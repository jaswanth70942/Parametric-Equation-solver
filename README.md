# Parametric-Curve-Solver

This repository contains a complete solution for recovering the unknown parameters **θ**, **M**, and **X** of a nonlinear parametric curve, using only a set of (x, y) data points. The challenge is that the dataset does **not** provide the parameter **t**, which the curve depends on — making this a difficult inverse problem. In this project, I've uses geometric reasoning and advanced optimization methods to accurately reconstruct both the hidden parameters and the underlying curve.

---

#  Final Recovered Parametric Equation

The best-fit model for the given dataset is:

\[
x(t)=t\cos(0.5278)-e^{0.03047|t|}\sin(0.3t)\sin(0.5278)+55.611
\]

\[
y(t)=42+t\sin(0.5278)+e^{0.03047|t|}\sin(0.3t)\cos(0.5278)
\]

These equations reproduce the dataset with very high accuracy.

---

#  Recovered Parameters

| Parameter | Value |
|----------|--------|
| **θ** | 30.23° (0.5278 rad) |
| **M** | 0.030474 |
| **X** | 55.611 |
| **Final L1 Loss** | **0.4065**  |

An L1 error near **0.40** indicates an extremely close match between the model and the actual curve.

---

# Solution to the Problem & Explanation

Here is a step-by-step breakdown of the process followed to find these parameters.

---

## **1. Understanding the Parametric Model**

The curve is defined by:

- a linear drift component (involving **t**)  
- an oscillatory component (involving **exp(M|t|) * sin(0.3t)**)  
- a rotation by angle **θ**  
- a horizontal shift **X**

But the dataset gives only **(x, y)** — not **t**.

So the primary task is to recover both:
- the unknown parameters (θ, M, X)
- the missing t-values for each data point

---

## **2. Observing the Rotation Structure**

While rewriting the equations, we find:

\[
(x', y') = \text{Rotation}_\theta \big( t,\; v(t) \big)
\]

where:
- \(x' = x - X\)
- \(y' = y - 42\)
- \(v(t) = e^{M|t|}\sin(0.3t)\)

This means each data point is just a **rotated** version of \((t, v)\).

This geometric insight makes the problem solvable.

---

## **3. Undoing the Rotation (Inverse Mapping)**

To recover t for each data point:

\[
t_{\text{calc}} = x'\cos\theta + y'\sin\theta
\]

\[
v_{\text{calc}} = -x'\sin\theta + y'\cos\theta
\]

If θ and X are correct, these will align with the true mathematical form.

---

## **4. Matching the Oscillation Term**

The model expects:

\[
v_{\text{expected}} = e^{M|t_{\text{calc}}|}\sin(0.3t_{\text{calc}})
\]

So I've defined an error between:

- what the geometry gives \(v_{\text{calc}}\)
- what the parametric model predicts \(v_{\text{expected}}\)

This becomes the key quantity to minimize.

---

## **5. Defining the Optimization Objective**

I've minimized:

\[
\sum_i \left( v_{\text{calc},i} - v_{\text{expected},i} \right)^2
\]

The correct parameters will make both sides match closely.

---

## **6. Enforcing Problem Constraints**

The problem specifies:

- \(0° < θ < 50°\)
- \(-0.05 < M < 0.05\)
- \(0 < X < 100\)
- Valid t-values must lie within \(6 < t < 60\)

Penalties and bounds ensure the optimizer respects these restrictions.

---

## **7. Global + Local Optimization Strategy**

I've used a two-stage optimization:

1. **Differential Evolution**  
– finds a good global region  

2.  **L-BFGS-B & Powell**  
– refine the parameters precisely  

This combination avoids poor local minima and improves accuracy.

---

## **8. Reconstructing and Refining t-values**

After a first approximation:

1. Each data point is assigned its most likely t-value  
2. These t-values are smoothed using cubic splines  
3. Parameters are re-optimized using better-aligned t  
4. Repeat iteratively  

This process dramatically improves accuracy by fixing phase misalignment.

---

## **9. Final Result**

After several refinement rounds, the solver finally converges to:

1. A very accurate reconstruction  
2. L1 error ≈ **0.4065**  
3. A visually perfect match between model and data  

This confirms that the recovered parameters correctly describe the dataset.

---

# Code

The | **solve.py** | file contains the Python script used to perform this optimization.

## **Dependencies**
1. Python 3x
2. pandas
3. numpy
4. matplotlib
5. scipy
6. sklearn

