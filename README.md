# HyperTuneDOE


# Machine Learning Model Hyperparameter Tunning using Design of Experiments and Response Surface Modeling

## Abstract
Design of experiments (DOE) is a branch of applied statistics that is used for
planning, executing, analyzing and interpreting experiments. DOE gives a frame
work that enables the evaluation of the effects of each input factor over a target
variable. It is widely used and very common in research practices but there is not
much available literature in which this methods are used to understand the interac-
tions between hyper-parameters when tuning machine learning models. This paper
presents a methodology to study the interactions between hyper-parameters of a
decision tree regressor. The goal is to analyze the main effects and interactions
between selected hyper-parameters with factorial designs. This knowledge is used to
tune the decision tree by modeling the response surface between the experiments
and the target variables. The benefits of this approach includes fewer training runs
when comparing to common practices such as grid searching.

## Methodology
In this paper, a design of experiments (DOE) methodology is proposed to screen out
the most significant hyper-parameters of a simple machine learning model. Then as the
next step, to fine tune the model, a response surface modeling (RSM) technique is used
to approximate the models performance with different values of hyper-parameters. A
fractional factorial design is used to get an idea of the most significant factors. This helps
to reduce the number of runs in the full factorial design that follows. In a realistic setting
when a model has to be trained on a great amount of data, understanding what the most
significant parameters are can reduce the error of the model considerably in a smaller
time span. The full factorial design is done to break all the confounding effects of the
fractional factorial and get a clear understanding of the existing interaction effects. As
a second phase, two rounds of a RSM designs are used to approximate the interactions
between the hyper-parameters in the proximity of a centerpoint. At the end, a fine tuned
model is presented and compared to an optimized model with classical grid searching.
