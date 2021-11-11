# Bayesian Mixture Modeling with Wasserstein Distance

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

## Overview

Mixture Models typically find application in density estimation and clustering problems.
Conventional Bayesian posterior inference on cluster-specific parameters struggles when clusters are placed too close to each other.

As an alternative, Repulsive Mixture Models generate the components from a repulsive process that naturally favours separation of clusters.

<p align="center">
    <img src="https://user-images.githubusercontent.com/91596609/141163222-2d8e2893-cc11-42a7-9f10-31366280264e.png" width="400" alt="Scenario"/>
</p>

We want to build a Repulsive Mixture Model by adding to the prior of the centers a penalization term proportional to the Wasserstein distance.

Our goals as we proceed with the project are:
* formalize a Repulsive model using the Wasserstein Distance 
* investigate other models:
   * with heavier tails like t-student, but where the Wasserstein Distance can still be expressed in closed form
   * skew-normal, since it’s not a location-scale distribution and so the Wasserstein Distance can’t be expressed in closed form
