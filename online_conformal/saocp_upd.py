from typing import Dict, Tuple

import numpy as np
import pandas as pd

from online_conformal.base import BasePredictor
from online_conformal.enbpi import EnbMixIn
from online_conformal.utils import pinball_loss, pinball_loss_grad, Residuals


class _OGD:
    def __init__(self, t, scale, alpha, yhat_0, g=8):

        # Scale-free online gradient descent parameters
        self.scale = scale
        self.base_lr = scale / np.sqrt(3)
        self.alpha = alpha
        self.yhat = yhat_0
        self.grad_norm = 0

        # Meta-algorithm parameters
        u = 0
        while t % 2 == 0:
            t /= 2
            u += 1
        self.lifetime = g * 2**u
        self.z = 0  # sum of differences between losses & meta-losses
        self.wz = 0  # weighted sum of differences between losses & meta-losses
        self.s_t = 0  # how long the predictor has been alive

    @property
    def expired(self):
        return self.s_t > self.lifetime

    def loss(self, y):
        return pinball_loss(y, self.yhat, 1 - self.alpha)

    @property
    def w(self):
        return 0 if self.s_t == 0 else self.z / self.s_t * (1 + self.wz)

    def update(self, y, meta_loss):
        # Update meta-algorithm weights
        w = self.w
        g = np.clip((meta_loss - self.loss(y)) / self.scale / max(self.alpha, 1 - self.alpha), -1 * (w > 0), 1)
        self.z += g
        self.wz += g * w
        self.s_t += 1

        # Update estimator
        grad = pinball_loss_grad(y, self.yhat, 1 - self.alpha)
        self.grad_norm += grad**2
        if self.grad_norm != 0:
            self.yhat = max(0, self.yhat - self.base_lr / np.sqrt(self.grad_norm) * grad)


class SAOCP_UPD(BasePredictor):


    def __init__(self, *args, horizon=1, max_scale=None, n_model = 1, lifetime=8, **kwargs):
        self.n_model = n_model
        self.t = 1
        if max_scale is None:
            self.scale = {}
        else:
            self.scale = {h + 1: max_scale for h in range(horizon)}
        self.experts = {h + 1: {} for h in range(horizon)}
        self.lifetime = lifetime
        super().__init__(*args, horizon=horizon, **kwargs)

        residuals = self.residuals
        self.residuals = {}
        resmodel = {}
        for n in range(n_model):
            resmodel[n+1] = residuals
            self.residuals[n+1] = Residuals(self.horizon)

        for h in range(1, self.horizon + 1):
            rr = []
            for n in range(n_model):
                r = resmodel[n+1].horizon2residuals[h]
                if h not in self.scale:
                    self.scale[h] = 1 if len(r) == 0 else np.max(np.abs(r)) * np.sqrt(3)
                rr.append(pd.Series(r , dtype = float))
            self.update(rr, pd.Series([np.zeros(len(rr[0]))]*len(rr)), h)

    def find_best_model(self, horizon):
        experts = self.experts[horizon]
        best_exp = {key: None for key in experts.keys()}
        best_exp_w = {key: None for key in experts.keys()}
        models_prob = {key: 0 for key in range(self.n_model)}
        for k in experts.keys():
            expert = experts[k]
            prior_w = {n: max(0,exp.w) for n, exp in expert.items()}
            sum_w = sum(prior_w.values())
            if sum_w != 0:
                probs = {n: w/sum_w for n,w in prior_w.items()}
            else: 
                probs = {n+1: 1/self.n_model for n in range(self.n_model)}
            best_exp[k] = sum(probs[n] * exp.yhat for n, exp in expert.items())
            best_exp_w[k] = sum(probs[n] * max(0,exp.w) for n, exp in expert.items())
            for kp in models_prob.keys():
                models_prob[kp] += probs[kp+1]
        sum_p = sum(models_prob.values())
        if sum_p != 0:
            models_prob  = {n+1: models_prob[n] / sum_p for n in range(self.n_model)}
        else:
            models_prob = {n+1: 1/self.n_model for n in range(self.n_model)}
        return best_exp , best_exp_w , models_prob



    def predict(self, horizon) -> Tuple[float, float]:
        experts , experts_w , models_prob= self.find_best_model(horizon)
        prior = {t: 1 / (t**2 * (1 + np.floor(np.log2(t)))) for t in experts.keys()}
        z = sum(prior.values())
        prior = {t: v / z for t, v in prior.items()}
        p = {t: prior[t] * experts_w[t] for t in experts.keys()}
        sum_p = sum(p.values())
        p = {t: v / sum_p for t, v in p.items()} if sum_p > 0 else prior
        delta = sum(p[t] * experts[t] for t in experts.keys())
        return -delta, delta, models_prob

    def create_expert(self, horizon, s_hat):
        return _OGD(self.t, self.scale[horizon], 1 - self.coverage, g=self.lifetime, yhat_0=s_hat)

    def update(self, ground_truth, forecast: pd.Series, horizon: int):
        residuals = {}
        for n in range(self.n_model):
            residuals[n+1] = np.abs(ground_truth[n] - forecast[n])
            self.residuals[n+1].extend(horizon, residuals[n+1].tolist())

        if horizon not in self.scale:
            return


        experts = self.experts[horizon]
        cnt = 1 if isinstance(residuals[1], np.float64) else len(residuals[1])
        for i in range(cnt):
            #remove and create expert
            s_hat = self.predict(horizon)[1]
            [experts.pop(t) for t in [k for k, v in experts.items() if v[1].expired]]
            new_exp = {}
            for n in range(self.n_model):
               new_exp[n + 1] = self.create_expert(horizon, s_hat)  

            experts[self.t] = new_exp


            s_hat_upd = self.predict(horizon)[1]
            meta_loss = []
            for n in range(self.n_model):
                meta_loss_m = pinball_loss(residuals[n+1] if isinstance(residuals[n+1], np.float64) else residuals[n+1][i],s_hat_upd, self.coverage)
                meta_loss.append(meta_loss_m)
            
            
            for expert in experts.values():
                for ex in range(len(expert.keys())):
                    expert[ex+1].update(residuals[ex+1] if isinstance(residuals[ex+1], np.float64) else residuals[ex+1][i] , meta_loss[ex])
            self.t += 1


class EnbSAOCP(EnbMixIn, SAOCP_UPD):
    pass
