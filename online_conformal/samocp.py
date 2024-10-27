from typing import Dict, Tuple

import numpy as np
import pandas as pd

from online_conformal.base import BasePredictor
from online_conformal.utils import pinball_loss_new,pinball_loss_grad_new


class _OGD_new:
    def __init__(self, t, alpha, alpha_0,w_0, g=8):

        self.base_lr = 1 / 20
        self.coverage = 0.9
        self.eps = 0.9
        self.alpha = alpha
        self.alpha_pred = alpha_0
        self.grad_norm = 0

        u = 0
        while t % 2 == 0:
            t /= 2
            u += 1
        self.lifetime = g*2**u
        self.weight =  w_0 
        self.s_t = 0  # how long the predictor has been alive

    @property
    def expired(self):
        return self.s_t > self.lifetime

    def loss(self, alpha_s):
        return pinball_loss_new(alpha_s, self.alpha_pred, self.alpha)
    
    def get_w(self):
        return self.weight
    
    def get_alpha(self):
        return self.alpha_pred

    @property
    def w(self):
        return 0 if self.s_t == 0 else self.weight

    def update(self, alpha_in):

        self.s_t += 1
        self.weight = self.weight * np.exp(-self.eps * self.loss(alpha_in))

        # Update estimator
        grad = pinball_loss_grad_new(alpha_in, self.alpha_pred, self.alpha)
        self.grad_norm += grad**2
        if self.grad_norm != 0:

            self.alpha_pred = self.alpha_pred - self.base_lr / np.sqrt(self.grad_norm) *grad

            


class SAMOCP(BasePredictor):

    def __init__(self, *args, horizon=1, max_scale=None, n_model = 1, lifetime=8, **kwargs):
        self.n_model = n_model
        self.t = 1
        self.experts = {h + 1: {} for h in range(horizon)}
        self.lifetime = lifetime
        self.wt = {}
        super().__init__(*args, horizon=horizon, **kwargs)

        new_exp = {}
        for n in range(self.n_model):
            new_exp[n + 1] = self.create_expert(horizon, 1 - self.coverage, 1/n_model) 

        self.experts[1][self.t] = new_exp
        self.wt[self.t] = min(0.9, 140/np.sqrt(self.lifetime*self.t))
        self.t += 1


    def find_best_model_t(self, horizon, t):
        experts = self.experts[horizon][t]
        probs = {n: expert.get_w() for n, expert in experts.items()}
        alphas = {n: expert.get_alpha() for n, expert in experts.items()}
        sum_w = sum(probs.values())
        probs = {n: w/sum_w for n,w in probs.items()}
        alpha_t = np.dot(list(probs.values()), list(alphas.values()))
        return alpha_t, probs
    

            
    def find_best_model(self, horizon):
        experts = self.experts[horizon]
        best_exp = {key: None for key in experts.keys()}
        best_exp_w = {key: self.wt[key] for key in experts.keys()}
        sum_exp_w = sum(best_exp_w.values())
        best_exp_w = {n: w/sum_exp_w for n,w in best_exp_w.items()}
        models_prob = {key+1: 0 for key in range(self.n_model)}

        for k in experts.keys():
            alpha_t , probs_t = self.find_best_model_t(horizon, k)
            best_exp[k] = alpha_t
            for kp in models_prob.keys():
                models_prob[kp] += best_exp_w[k] * probs_t[kp]

        alpha = np.dot(list(best_exp_w.values()), list(best_exp.values()))

        return alpha, models_prob


    def predict(self, horizon) -> Tuple[float, float]:
        alpha_pred, models_prob = self.find_best_model(horizon)
        return alpha_pred, models_prob

    def create_expert(self, horizon, s_pred, w_0):
        return _OGD_new(self.t, 1 - self.coverage, g=self.lifetime, alpha_0=s_pred, w_0=w_0)

    def update(self, ground_truth, forecast: pd.Series, horizon: int):
        residuals = {}
        for n in range(self.n_model):
            residuals[n+1] = np.abs(ground_truth[n] - forecast[n])

        experts = self.experts[horizon] 

        #pop expired expert
        expired_experts = [k for k, v in experts.items() if v[1].expired]
        for t in expired_experts:
            experts.pop(t)
            self.wt.pop(t)

        #creat new expert
        new_exp = {}
        alpha_pred, model_probs = self.predict(horizon)
        for n in range(self.n_model):
            new_exp[n + 1] = self.create_expert(horizon, alpha_pred, model_probs[n+1])  
        experts[self.t] = new_exp

        new_var = self.t
        uu = 0
        while new_var % 2 == 0:
            uu += 1
            new_var /= 2

        self.wt[self.t] = min(0.9,140/np.sqrt(self.lifetime*2**uu))


        #update learning models of each expert
        for experts_t in experts.values():
            for exp in range(len(experts_t.keys())):
                experts_t[exp+1].update(residuals[exp+1])
        

        #update experts weights
        alpha_pred, probs = self.predict(horizon)
        alpha_opt = np.dot(list(residuals.values()),list(probs.values()))
        learner_loss = pinball_loss_new(alpha_opt, alpha_pred, 1 - self.coverage)
        for learner_t in experts.keys():
            alpha_pred_t, probs_t = self.find_best_model_t(horizon, learner_t)
            alpha_s = np.dot(list(residuals.values()),list(probs_t.values()))
            loss_t = pinball_loss_new(alpha_s, alpha_pred_t, 1 - self.coverage)
            r_t = learner_loss - loss_t
            uu = 0
            temp_learner_t = learner_t 
            while temp_learner_t % 2 == 0:
                uu += 1
                temp_learner_t /= 2
      
            eta_t = min(0.9, 140/np.sqrt(self.lifetime*2**uu))
            self.wt[learner_t] = self.wt[learner_t] * np.exp(-eta_t * r_t)


        self.t += 1


