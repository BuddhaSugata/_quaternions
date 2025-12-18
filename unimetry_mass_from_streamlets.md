# Mass as a functional of streamlet structure in Unimetry

*(draft derivation in the “streamlet + second moment” picture, without using mass to define the angle \(\zeta\))*


## 1. Streamlets, normalization and second moments

Consider a massive object as an ensemble of streamlets indexed by \(a\) with weights \(w_a\), \(\sum_a w_a = 1\).

Each streamlet has a normalized proto–flow direction \(\tilde{\boldsymbol\chi}_a\) in the proto–space \(\mathcal E\), decomposed with respect to the **object’s own temporal axis** \(\hat e_\tau\) and a spatial triad \(\{\hat e_i\}\):

\[
\tilde{\boldsymbol\chi}_a
 = \tilde X^0_a\,\hat e_\tau + \tilde{\mathbf X}_a,
 \qquad
 \|\tilde{\boldsymbol\chi}_a\|^2
 = (\tilde X^0_a)^2 + \|\tilde{\mathbf X}_a\|^2
 = 1.
\]

We parameterize each streamlet by an angle \(\Theta_a\) between its flow and the temporal axis:

\[
\tilde X^0_a = \cos\Theta_a,\qquad
\|\tilde{\mathbf X}_a\| = \sin\Theta_a.
\]

The **rest configuration** of the object is defined as that frame in which the *first spatial moment* vanishes:

\[
\sum_a w_a \tilde{\mathbf X}_a = 0.
\]

However, the **second moment** in general does not vanish and contains structural information about the object.

Define the second moments:

\[
T_B := \sum_a w_a \cos^2\Theta_a,
\qquad
\mathbf C_B := \sum_a w_a \sin^2\Theta_a\;\mathbf u_a\otimes\mathbf u_a,
\]
where \(\mathbf u_a := \tilde{\mathbf X}_a / \|\tilde{\mathbf X}_a\|\) are unit spatial directions (when \(\sin\Theta_a\neq 0\)).

From \(\cos^2\Theta_a + \sin^2\Theta_a = 1\) and \(\sum_a w_a = 1\) it follows that

\[
T_B + \operatorname{tr}\mathbf C_B = 1.
\]

Here:

- \(T_B\) encodes the **temporal share of total flow “budget”**,
- \(\mathbf C_B\) encodes both **amount** and **shape** of the spatially looped flow.


## 2. Structural angle \(\zeta\) from the second moment (without mass)

We now define a **purely structural** “budget angle” \(\zeta\) by

\[
T_B = \cos^2\zeta,\qquad
\operatorname{tr}\mathbf C_B = \sin^2\zeta.
\]

This is **not** defined using any notion of physical mass or charge.  
It is determined solely by:

- the ensemble of streamlets \(\{\tilde{\boldsymbol\chi}_a, w_a\}\),
- the choice of the object’s own temporal axis \(\hat e_\tau\),
- and the second moment data \((T_B, \mathbf C_B)\).

Thus \(\zeta\) is a **geometric property of the proto–flow structure**, not a function of the observed rest mass.

Interpretation:

- \(\cos^2\zeta\) is the fraction of the unit flow magnitude that runs “along time” (along \(\hat e_\tau\));
- \(\sin^2\zeta\) is the fraction trapped in spatial loops and anisotropies encoded by \(\mathbf C_B\).


## 3. Total proto–flow magnitude and temporal share

We now introduce a convenient normalization of the total proto–flow:

\[
\tilde H := \Big\langle\|\tilde{\boldsymbol\chi}_a\|^2\Big\rangle^{1/2}
         = \left(\sum_a w_a\,1\right)^{1/2}
         = 1.
\]

So the **total flow magnitude** of the object is set to unity; all nontrivial physics is in how this unit is split between temporal and spatial directions.

The **temporal share** of the flow, in the object’s rest configuration, is then

\[
\tilde X_0^2 := T_B = \cos^2\zeta,
\]

while the spatial share is \(\operatorname{tr}\mathbf C_B = \sin^2\zeta\).

After the standard SR calibration, where the physical flow is fixed as \(\boldsymbol\chi = c\,\widehat{\boldsymbol\chi}\) (so that \(\|\boldsymbol\chi\|=c\)), the temporal component of the physical flow becomes

\[
\chi_\tau^{(\text{phys})} = c\,\tilde X_0 = c\cos\zeta.
\]

In this sense, for the object itself the **effective “speed of light”** that controls the rate of its proper time is the temporal share \(c\cos\zeta\); the remaining \(c\sin\zeta\) is locked in spatial circulation.


## 4. Phase parameter \(\chi\), cyclic time \(\tau\), and invariant flow \(H\)

Introduce the **global phase parameter** \(\chi\) and the **local proper time** \(\tau\) of the object.  

By construction of the streamlet picture, each full cycle of the object’s internal flow corresponds to a full phase \(2\pi\) in \(\chi\), while \(\tau\) parametrizes the local cycles of the object’s internal clock.

We postulate that the rate at which the phase winds with respect to the local time is proportional to the **temporal share of the flow**:

\[
\frac{d\chi}{d\tau}
:= \dot\chi
 = k\,\tilde X_0^2
 = k\,\cos^2\zeta,
\]
where \(k\) is a **geometric constant of the object**, e.g. the ratio of the radii of two conjugate cycles \(k = R_1/R_2\) in the cyclic picture.

This is a key step:

- we are **not** inserting the rest mass here;
- we are using only the structural angle \(\zeta\) extracted from the second moment of the streamlets.

Now define \(H(\chi)\) as the **accumulated flow quantity** along the phase:

\[
\frac{dH}{d\chi} := \tilde H.
\]

With \(\tilde H \equiv 1\), this reduces to

\[
\frac{dH}{d\chi} = 1
\quad\Rightarrow\quad
H(\chi) = \chi + \text{const}
\sim \chi.
\]

Differentiating with respect to the local time:

\[
\dot H := \frac{dH}{d\tau}
        = \frac{dH}{d\chi}\frac{d\chi}{d\tau}
        = 1\cdot \dot\chi
        = k\,\cos^2\zeta.
\]

Thus the instantaneous **scalar flow rate** in the proper frame of the object is

\[
\boxed{
\dot H = k\,\tilde X_0^2
       = k\,\cos^2\zeta.
}
\]

Its square gives the invariant quadratic form in the proper frame:

\[
dH^2 = \dot H^2\,d\tau^2
     = k^2\cos^4\zeta\,d\tau^2.
\]

This quantity is **invariant under D-boosts**: D-rotations of the observer do not change the internal \(\zeta\), nor the intrinsic cyclic geometry encapsulated in \(k\). They only change the observed decomposition of 4-velocity, not the internal flow rate \(\dot H\).


## 5. Mass as a functional of the spatial second moment \(\mathbf C_B\)

We now address the central physical ansatz:

> **Rest mass should arise purely from the amount and structure of spatial streamlet projections.**

The temporal vs spatial split obeys

\[
T_B + \operatorname{tr}\mathbf C_B = 1
\quad\Longleftrightarrow\quad
\tilde X_0^2 + \operatorname{tr}\mathbf C_B = 1.
\]

A purely temporal object (\(\operatorname{tr}\mathbf C_B = 0\)) has \(\tilde X_0^2 = 1\): all flow goes along time, no spatial loops – a **massless** (“photon-like”) configuration.

A massive object necessarily has \(\operatorname{tr}\mathbf C_B > 0\): some part of the flow is trapped in spatial circulation, reducing the effective temporal share.

A natural way to extract a **scalar mass** from \(\mathbf C_B\) is to consider invariants of its spectrum. For example:

\[
I_1 := \operatorname{tr}\mathbf C_B = \sin^2\zeta,
\quad
I_2 := \operatorname{tr}(\mathbf C_B^2),
\quad
I_3 := \det\mathbf C_B.
\]

Then define the **rest mass** as

\[
m_0 := \kappa_*\,F(I_1,I_2,I_3),
\]
where:

- \(\kappa_*\) is a universal scale factor,
- \(F\) is a dimensionless structural functional such that:
  - \(F(0,\cdot,\cdot)=0\) (no spatial content \(\Rightarrow\) no mass),
  - \(F\) is monotone in \(I_1\),
  - dependence on \(I_2,I_3\) encodes shape/anisotropy.

The **simplest isotropic ansatz** (to illustrate the idea) is

\[
\boxed{
m_0(\zeta,\mathbf C_B)
 := \kappa_*\,\sqrt{\operatorname{tr}\mathbf C_B}
 = \kappa_*\,\sin\zeta.
}
\]

- If \(\operatorname{tr}\mathbf C_B = 0\), then \(m_0 = 0\): no spatial loops, no mass.
- For more complex objects, \(\operatorname{tr}\mathbf C_B\) grows with the amount of spatial circulation in the streamlet ensemble, and so does \(m_0\).

More refined models can take e.g.

\[
m_0 = \kappa_*\,(\operatorname{tr}\mathbf C_B)^{3/2}
    = \kappa_*\,\sin^3\zeta,
\]

or other combinations of \(I_1,I_2,I_3\), but the **key structural statement** is:

> Rest mass is a scalar functional of the spatial second moment \(\mathbf C_B\), i.e. of the **amount and geometry of spatial streamlet projections**,  
> not an input parameter used to define \(\zeta\).


## 6. Energy and D-boosts: recovering \(E = \gamma m_0 c^2\)

In Energy.tex, the phase–space energy density is written as

\[
e(\chi) := \kappa(\chi)\,\dot H^3,
\]

and the integrated phase–energy over a phase slice \(\Sigma_\chi\) is

\[
E_\chi := \int_{\Sigma_\chi} e\,dV_\chi
       \equiv m_0 c^2.
\]

Here \(\kappa(\chi)\) is a structural coefficient depending only on internal configuration; \(\dot H\) is calibrated such that in the SR limit \(\dot H \equiv c\).

In the present picture, we decompose

\[
\kappa = \frac{1}{c}\,m_0(\zeta,\mathbf C_B)
      = \frac{\kappa_*}{c}\,F(I_1,I_2,I_3).
\]

Thus

\[
E_\chi = m_0(\zeta,\mathbf C_B)\,c^2
\]

is a **scalar invariant** fully determined by the streamlet structure.

Now consider a D-boost (Lorentz-like D-rotation) between the object’s rest frame and an external observer’s frame. By construction:

- the internal quantities \(\zeta, \mathbf C_B, k, \dot H\) are invariant under D-boosts,
- only the decomposition of 4-velocity into temporal/spatial parts changes.

Under a D-boost with rapidity \(\theta\) (velocity \(v = c\sin\theta\), factor \(\gamma = \sec\theta\)), the energy measured by the external observer is

\[
E = \gamma\,E_\chi
  = \gamma\,m_0(\zeta,\mathbf C_B)\,c^2.
\]

Therefore, the **standard relativistic energy formula**

\[
\boxed{
E = \gamma m_0 c^2
}
\]

emerges with:

- \(m_0\) derived **purely from the spatial structure of the streamlet ensemble** via \(\mathbf C_B\) and \(\zeta\),
- \(c^2\) as the invariant scale set by the calibrated flow rate \(\dot H\),
- \(\gamma\) as the familiar boost factor coming solely from the D-rotation between temporal axes, with no change in the internal structure.

In this way, Unimetry interprets mass as a **measure of how much of the proto–flow is locked into spatial cycles**, while retaining the standard relativistic energy–momentum relations on the observable side.
