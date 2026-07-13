# Multitask

This repository implements a **multitask learning framework** based on **Gaussian Process Regression (GPR)**, **SOAP descriptors**, and **Multitask Gaussian Processes**.

The main objective of this project is to perform regression from molecular representations generated using **SOAP (Smooth Overlap of Atomic Positions)** descriptors combined with Gaussian Process models. The approach is based on the combination of the **GAP (Gaussian Approximation Potentials)** framework with **Multitask Gaussian Processes**, allowing the model to exploit correlations between multiple related tasks.

## Main Components

- **Gaussian Process Regression (GPR)**  
  A non-parametric Bayesian regression method used to model complex relationships between input representations and target properties.

- **SOAP Representation**  
  Molecular structures are encoded using SOAP descriptors, providing rotationally and translationally invariant representations suitable for atomistic machine learning applications.

- **Multitask Gaussian Process**  
  An extension of standard Gaussian Processes that jointly learns multiple correlated tasks, enabling information transfer between different prediction targets.

## Overview

The workflow consists of the following steps:

1. Generating SOAP descriptors from atomic structures.
2. Selecting the most informative points.
3. Optimising the model hyperparameters.
4. Constructing the covariance matrix.
5. Performing predictions using the trained Gaussian Process model.

---

# Project Structure

The project is organised into two main sections:

- **Data shaping (`read_data`)**  
  Responsible for preparing and formatting the input data.

- **Multitask framework (`multitask`)**  
  Handles the complete Gaussian Process workflow, from data management to model construction and predictions.

---

# Data Shaping

Starting from `.xyz` datasets, the `read_data` function prepares the data in the appropriate format required for the subsequent steps.

The function separates the input features and target outputs into two distinct objects.

A feature normalisation step is also included. This procedure currently serves as a proof of concept and requires further investigation to evaluate its impact on model performance.

---

# `multitask` Function

The `multitask` function contains the core implementation of the Gaussian Process framework.

## 1. Sparse Point Selection

The first step consists of creating a sparse representation of the covariance matrix using the `create_filter_cov` function.

The selection of representative points is performed using a **pivoted QR decomposition algorithm**, which identifies the most informative rows of the global covariance matrix.

## 2. Hyperparameter Optimisation

The second step consists of optimising the model hyperparameters by maximising the likelihood function.

The optimisation relies on:

- **L-BFGS** optimisation algorithm
- **Automatic differentiation (autodiff)**

These tools allow efficient computation of gradients during the optimisation process.

## 3. Covariance Matrix Construction

The covariance matrix is then constructed using the `inner_kernel.jl` module.

This step defines the similarity measure between molecular representations and is a key component of the Gaussian Process model.

## 4. Prediction

The final step performs predictions using the trained Gaussian Process model.

The output consists of:

- The predicted mean values
- The associated predictive variance

These quantities provide both the model prediction and an estimation of its uncertainty.