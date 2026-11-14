# LEARNING.md — ft_linear_regression

This file is the defense notebook for the 42 project **ft_linear_regression**.

**How to use it**

1. **Part A (this part)** is every concept you must own *before* writing code: definitions, formulas, worked numbers, traps, oral answers.
2. **Part B** is the implementation notebook: the file/step plan, then each step explained **before** we move on. We do not start step N+1 until step N is understood.

Re-read this entire file before evaluation. If you cannot reconstruct a formula or a decision from memory after reading it, that section is not done yet.

The dataset we will train on is `data/data.csv`: 24 cars, columns `km` (mileage) and `price`. Mileage runs roughly from 22 899 to 240 000; prices from 3 650 to 8 290.

---

# Part A — Concepts

---

## 1. What machine learning is (in this project)

The subject quotes Tom Mitchell:

> A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in **T**, as measured by **P**, improves with experience **E**.

Map that onto this project:

| Letter | Meaning here |
|--------|----------------|
| **T** (task) | Given a car’s mileage, output an estimated price |
| **E** (experience) | The 24 rows in `data.csv` |
| **P** (performance) | How close estimates are to real prices (we will use a cost `J`, then later MSE / MAE / R²) |

**Learning** here is not “the program memorizes 24 prices.” It is: the program **adjusts two numbers** (`θ0`, `θ1`) so that a **rule** `price ≈ θ0 + θ1 × mileage` gets better (lower error) as it sees the dataset.

**Model vs algorithm vs hypothesis**

- **Hypothesis** `hθ(x)`: the *shape* of the rule we assume (a straight line).
- **Parameters** `θ0`, `θ1`: the numbers that pick *which* line.
- **Learning algorithm**: how we search for good θ (here: **batch gradient descent**).
- **Model**: the trained hypothesis (the line after training, i.e. the saved θ).

If someone asks “did the computer understand cars?” the honest answer is **no**. It fitted a line to 24 points.

---

## 2. Supervised learning, regression, one feature

**Supervised learning:** each training example is a pair `(input, correct output)`. We have `x = km` and `y = price`. The algorithm is *supervised* because the CSV tells it the right prices.

**Unsupervised learning** (not this project): only inputs, no labels (e.g. clustering customers). No `y`.

**Regression vs classification**

- **Classification:** output is a category (spam / not spam). Discrete.
- **Regression:** output is a **continuous number** (price in whatever currency the CSV uses). This project is **regression**.

**Simple (univariate) linear regression:** **one** input feature (`mileage`) and a **linear** hypothesis. Multivariate linear regression would be several features (year, km, brand, …) and still a linear combination. We do **not** need that for the mandatory part.

**Training vs prediction**

- **Training:** look at all `(mileage, price)` pairs, update θ. Result: two numbers saved to disk.
- **Prediction:** do **not** look at the CSV prices. Take a mileage from the user, apply `hθ(x)`. If we never trained, θ are 0, so the estimate is 0.

---

## 3. The hypothesis (mandatory formula)

The subject **requires** this formula:

\[
\text{estimatePrice}(\text{mileage}) = \theta_0 + \theta_1 \cdot \text{mileage}
\]

Same thing in usual ML notation, with \(x\) = mileage and \(h_\theta\) = estimated price:

\[
h_\theta(x) = \theta_0 + \theta_1 x
\]

### What each theta means

- **θ0 (bias / intercept):** estimated price when mileage is **0**. On a plot, it is where the line crosses the **price axis**. For used cars, a real car at 0 km is not in the dataset; θ0 is still a valid mathematical intercept. It will often be **positive** and in the same ballpark as “expensive, low-km cars” (thousands, not zero).

- **θ1 (slope / weight):** how much estimated price **changes** when mileage increases by **1 km**.
  - If θ1 = **−0.02**, then 1 extra km lowers the estimate by 0.02 (same units as price).
  - 10 000 extra km would change the estimate by \(10000 \times \theta_1\).
  - For this dataset we **expect θ1 < 0**: more kilometers → cheaper car. If after training θ1 is clearly positive, the implementation or the scaling is wrong.

### Graph picture

Imagine a scatter plot: horizontal axis = km, vertical = price. Each CSV row is a **dot**. The hypothesis is **one straight line** through that cloud.

- A “good” line passes near most dots.
- A “bad” line misses them (all estimates too high, too low, or the slope is the wrong sign).

**Example (tiny, not the real CSV):**

| mileage `x` | price `y` |
|-------------|-----------|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

The perfect line is \(h(x) = 1 + 2x\) (θ0 = 1, θ1 = 2).  
Check: \(1 + 2\cdot1 = 3\), \(1 + 2\cdot2 = 5\), \(1 + 2\cdot3 = 7\).

If instead we use θ0 = 0, θ1 = 0: every estimate is 0. Errors are huge. That is **exactly** the starting point of the subject: *before training, θ0 and θ1 are 0*.

### What the line cannot do

If the true relationship were a curve (price collapses after 200 000 km, then flattens), a straight line can only be a **rough average trend**. The subject still wants a line. That is a **model assumption**, not a law of nature.

---

## 4. Residuals and the cost function `J`

The PDF does **not** write `J`, but gradient descent **is** “walk downhill on `J`.” Evaluators often ask for this.

### Residual (error on one example)

For example `i` (counting from 0 as in the PDF):

\[
\text{error}^{(i)} = h_\theta(x^{(i)}) - y^{(i)} = \text{estimatePrice}(\text{mileage}[i]) - \text{price}[i]
\]

- **Positive error:** we **overestimated** (line is above the point).
- **Negative error:** we **underestimated** (line is below the point).

**Do not mix the order.** The subject writes `(estimatePrice − price)`, not `(price − estimatePrice)`. If you flip the sign, gradient descent **climbs** the cost instead of descending (θ run the wrong way).

### Why not “just sum the errors”?

If one point is +100 and another is −100, the sum is 0: looks perfect, but both predictions are bad. So we **square** errors: both become +10000, they add up, and large mistakes are punished more than small ones.

### Mean squared error and `J`

Let **m** = number of training examples.

**m is 24** for `data.csv`. That is the “I let you guess what m is” in the subject.

Two common writings:

**Mean squared error (MSE)** — easy to say in an oral:

\[
\text{MSE} = \frac{1}{m} \sum_{i=0}^{m-1} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2
\]

**Cost used in many textbooks** (the extra `1/2` only makes the derivative prettier; it does **not** change *which* θ is the minimum):

\[
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=0}^{m-1} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2
\]

`J` is a **function of θ**, not of a single car. For each pair `(θ0, θ1)` you get **one number**: “how bad is this line on the whole dataset?”

- `J ≥ 0` always.
- `J = 0` only if the line hits every point (almost never with real data).
- Training = **minimize `J`**.

### Tiny numerical example

Use the three points `(1,3), (2,5), (3,7)` and θ0 = 0, θ1 = 0.

- Estimates: 0, 0, 0
- Errors: 0−3 = −3, 0−5 = −5, 0−7 = −7
- Squared: 9, 25, 49
- Sum = 83
- `m = 3`
- MSE = 83/3 ≈ 27.67
- `J` with 1/(2m) = 83/6 ≈ 13.83

If we plug the perfect θ0 = 1, θ1 = 2, every error is 0, `J = 0`.

### Picture of `J`

Think of a bowl (paraboloid) over the plane `(θ0, θ1)`. Height = `J`. Gradient descent starts at `(0, 0)` and takes small steps **downhill** until it sits near the bottom.

For *simple linear* regression, that bowl is **convex**: one global minimum, no “wrong valley.” That is why this algorithm is a safe first ML project. (Neural nets are not convex; this project is.)

---

## 5. Gradient descent — the training function

### Idea

The **gradient** of `J` is a 2D arrow: “which way does `J` increase the fastest?” We go the **opposite** way.

\[
\theta_j \leftarrow \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}
\]

- **α (alpha)** = **learning rate** (the subject’s `learningRate`). A positive small number we choose.
- The **minus** is mandatory: minus gradient = descent.

### Derivatives (why the PDF looks like it does)

Using \( J = \frac{1}{2m} \sum (h_\theta(x^{(i)}) - y^{(i)})^2 \) and \( h_\theta(x) = \theta_0 + \theta_1 x \):

\[
\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=0}^{m-1} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)
\]

\[
\frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=0}^{m-1} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr) \, x^{(i)}
\]

**That is exactly the subject’s sums**, then multiplied by `learningRate`:

\[
\text{tmp}\theta_0 = \alpha \cdot \frac{1}{m} \sum_{i=0}^{m-1} \bigl(\text{estimatePrice}(\text{mileage}[i]) - \text{price}[i]\bigr)
\]

\[
\text{tmp}\theta_1 = \alpha \cdot \frac{1}{m} \sum_{i=0}^{m-1} \bigl(\text{estimatePrice}(\text{mileage}[i]) - \text{price}[i]\bigr) \cdot \text{mileage}[i]
\]

So `tmpθ0` is \(\alpha \cdot \partial J / \partial \theta_0\), and same for θ1.

### The assignment the PDF does not write (you must know it)

The PDF shows **tmp**, then says: guess `m`, use the latest θ inside `estimatePrice`, **update θ0 and θ1 simultaneously**.

The actual parameter step is:

\[
\theta_0 \leftarrow \theta_0 - \text{tmp}\theta_0
\]
\[
\theta_1 \leftarrow \theta_1 - \text{tmp}\theta_1
\]

**If you set `θ0 = tmpθ0` instead of subtracting, you are not doing gradient descent.** tmp is a *step*, not the new θ.

### Simultaneous update

**Wrong:**

1. Compute tmpθ0 from current θ.
2. Set `θ0 = θ0 - tmpθ0` immediately.
3. Compute tmpθ1 using the **new** θ0 inside `estimatePrice`.

That uses two different lines in one iteration. The subject forbids that.

**Right:**

1. Using the **same** current `(θ0, θ1)`, compute **both** sums (both tmps).
2. **Then** apply both subtractions.

In code this is: two local variables `tmp0`, `tmp1`, then two assignments.

### Batch gradient descent

The sums run over **all m examples** every update. That is **batch** GD.

- **Stochastic GD:** update after **each** example (noisy path). Not what the formula says.
- **Mini-batch:** update after a small group. Also not the subject.

You should *know* those names; you should *implement* batch.

### One iteration vs one epoch

In this project, **one iteration** = one simultaneous update using the full dataset. People also call that **one epoch** (one full pass over the data). We repeat hundreds or thousands of times until `J` stops falling.

### Worked example — one iteration by hand

Dataset (not the CSV — small enough to compute without a calculator if needed):

| i | mileage \(x\) | price \(y\) |
|---|----------------|-------------|
| 0 | 1 | 3 |
| 1 | 2 | 5 |

- `m = 2`
- start: `θ0 = 0`, `θ1 = 0`
- pick `α = 0.1`

**Estimates:** \(h(1) = 0\), \(h(2) = 0\).

**tmpθ0:**

\[
\alpha \cdot \frac{1}{2} \bigl[ (0-3) + (0-5) \bigr] = 0.1 \cdot \frac{1}{2} \cdot (-8) = 0.1 \cdot (-4) = -0.4
\]

**tmpθ1:**

\[
\alpha \cdot \frac{1}{2} \bigl[ (0-3)\cdot 1 + (0-5)\cdot 2 \bigr] = 0.1 \cdot \frac{1}{2} \cdot (-3-10) = 0.1 \cdot (-6.5) = -0.65
\]

**Simultaneous subtract:**

\[
\theta_0 = 0 - (-0.4) = 0.4
\]
\[
\theta_1 = 0 - (-0.65) = 0.65
\]

New line: \(h(x) = 0.4 + 0.65x\).

Check: \(h(1) = 1.05\) (true 3), \(h(2) = 1.7\) (true 5). Still too low, but **closer than 0**. Next iterations keep climbing toward the true line \(1 + 2x\).

**Intuition for the signs:** all errors were negative (underestimate). The average error is negative, so tmpθ0 is negative, and **minus a negative** **increases** θ0. Same idea for θ1: the line needed a steeper positive slope.

On the **real** car data, errors at the start (θ = 0) are `0 − price` = **negative** (prices are thousands). So θ0 will **increase** from 0. For θ1, each term is `(negative error) × (large positive km)` = more negative tmpθ1, so θ1 = 0 − (negative) = **positive** at the very first step if you do **not** think carefully… Wait: error = estimate − price = 0 − 3650 = **negative**. negative × positive km = **negative** sum. tmpθ1 = α × (negative) = **negative**. θ1 ← 0 − (negative) = **positive**.

That first step can look confusing: **the first update may push θ1 positive**, then later updates (once θ0 is large enough that we **overestimate** cheap high-km cars) pull θ1 **negative**. After convergence we still expect **θ1 < 0**. If you only run **one** iteration on raw km, you have not finished learning. This is a good oral nuance.

(After **normalization**, the first steps are better behaved; see section 7.)

---

## 6. Learning rate, iterations, divergence

**α too small:** each step is tiny. `J` falls slowly. You wait forever. The line barely moves.

**α too large:** you overshoot the bottom of the bowl. `J` **oscillates** or **explodes** (θ become huge, then NaN). The plot of `J` vs iteration goes **up**.

**α “just right”:** `J` decreases smoothly and flattens.

**How we choose α in practice (no magic formula in the subject):** try values like `0.001, 0.01, 0.1, 1` **after** scaling (section 7). Watch `J` every 100 iterations. If `J` grows, **decrease α**. If `J` crawls, **increase α** a bit.

**When to stop**

- Fixed number of iterations (simple, easy to defend: “I ran 1000 steps; `J` had flattened”).
- Or stop when `|J_new − J_old| < ε` (e.g. 1e-8).

There is no unique “correct” α in the subject. There **is** a correct **update formula**. α is a **hyperparameter** (you choose it; it is not learned by the same GD loop unless you do extra work we do not need).

**Sanity check during training:** print `J`. It should **not** increase. If it does, α is too big **or** you flipped a sign **or** you forgot simultaneous update **or** you did not scale features.

---

## 7. Feature scaling (not in the PDF, often mandatory in practice)

### The problem

Mileage is ~10⁴–10⁵. Price is ~10³–10⁴.

Look at tmpθ1: you multiply **error** (thousands) by **mileage** (hundreds of thousands). One term can be on the order of \(5000 \times 200000 = 10^9\). Then you average and multiply by α.

- If α is `0.1` like in the toy example, θ1 jumps by a gigantic amount → **divergence**.
- If you pick α so small that θ1 is stable (e.g. `1e-10`), then tmpθ0 (which does **not** multiply by km) crawls like a snail. The two parameters live on **different scales**. GD becomes miserable.

This is why the subject says visualization helps debugging: you will **see** the line fly off the plot if you train on raw km with a naive α.

### What scaling does

We train on a **transformed** mileage whose values are small, typically around 0–1 or mean 0 and std 1.

**Standardization (z-score)** — common choice:

\[
x' = \frac{x - \mu}{\sigma}
\]

- \(\mu\) = mean of **mileage** on the training set
- \(\sigma\) = standard deviation of mileage (if σ = 0, all x equal: cannot scale; degenerate data)

After this, most `x'` are roughly in `[-2, 2]`. tmpθ0 and tmpθ1 have **similar magnitude**. α like `0.1` can work.

**Min-max scaling:** \( x' = (x - x_{\min}) / (x_{\max} - x_{\min}) \) maps km into `[0, 1]`. Also valid. Be ready to explain whichever we implement.

You *can* also scale `y` (price). It is optional. If you scale `y`, you must invert that scale when reporting prices. The subject’s predict formula outputs **price**, not a normalized price. Simplest defense: **scale only x (mileage), leave y in real money.**

### The trap: predict must stay `θ0 + θ1 * mileage`

The subject’s predict program uses **raw** mileage, not `x'`.

If you train with `h = θ0' + θ1' * x'`, those θ' are **not** the θ you save for predict.

**Algebra to convert back (standardization):**

\[
h = \theta_0' + \theta_1' \cdot \frac{x - \mu}{\sigma} = \underbrace{\left(\theta_0' - \theta_1'\frac{\mu}{\sigma}\right)}_{\theta_0} + \underbrace{\left(\frac{\theta_1'}{\sigma}\right)}_{\theta_1} \cdot x
\]

Save **θ0** and **θ1** (raw space). Predict is again exactly:

\[
\text{estimatePrice}(x) = \theta_0 + \theta_1 x
\]

If an evaluator asks “did you cheat the formula?” you say: **GD ran on scaled x for numerical stability; we algebraically rewrote the same line in unscaled x; the hypothesis in predict is the one in the PDF.**

**Never** save only θ' and then forget μ, σ in predict — unless you also save μ, σ and apply them at predict time, which makes the visible formula look different from the PDF. **Denormalizing θ is the cleaner 42 story.**

### Mean and std — you must define them

For a list \(x_0, \ldots, x_{m-1}\):

\[
\mu = \frac{1}{m} \sum_i x_i
\]

\[
\sigma = \sqrt{\frac{1}{m} \sum_i (x_i - \mu)^2}
\]

(Some people use `m-1` in the variance. For scaling GD it barely matters; pick one and stick to it. Population `1/m` is fine.)

---

## 8. The two programs and persistence

**Program 1 — predict**

- Prompt the user for a mileage.
- Load θ0, θ1 from a file written by train.
- Print `θ0 + θ1 * mileage`.
- **If the file does not exist** (train never ran): θ0 = θ1 = 0 → estimate **0**.

**Program 2 — train**

- Read `data.csv`.
- Run batch GD (with scaling as above).
- Write θ0, θ1 to disk.

The only coupling between the two programs is **that file**. Do not hardcode trained numbers inside `predict.py`.

**Why a file?** Two separate executables, as the subject asks. Training can take a moment; prediction should be instant and should still work tomorrow without retraining, as long as the theta file is there.

---

## 9. Closed-form solution (know it; do not use it as the trainer)

For simple linear regression there is an analytic minimum (normal equation / least squares). For one feature it is equivalent to “best fit line” formulas you may have seen in statistics.

**The subject wants gradient descent**, not `numpy.polyfit`, not sklearn, not “I inverted a matrix.” Using the closed form as the *official* training program is **cheating** under the peer-eval rules (“absence of libraries that do the work for you”, “use of the specified training function”).

You may *mention* that a closed form exists and that GD is the method we practice because it **generalizes** to models where there is no nice formula (neural nets).

---

## 10. Forbidden vs allowed tools

**Forbidden as the way you obtain θ**

- `numpy.polyfit`
- `sklearn.linear_model.LinearRegression` (and similar)
- `scipy.stats.linregress` as the trainer
- Excel / Numbers trendline copied into the program
- Any library call whose job is “fit a line”

**Allowed**

- Reading a CSV (`csv` module, or pandas **only to load tables**)
- Lists / arrays to store km and price
- `matplotlib` to **plot** (subject encourages visualization)
- numpy for **array math you still wrote as the PDF sums** — if we use it, we must still be able to rewrite the loop with plain Python. Safer for 42: **plain Python loops for the sums**, matplotlib for plots.

**Rule of thumb:** if the library **gives you θ**, it is cheating. If it **draws dots** or **parses a file**, it is fine.

---

## 11. Bonuses (only graded if mandatory is perfect)

### 11.1 Plot the data

Scatter: each car is a point `(km, price)`. You should see a **downward** cloud (higher km, lower price), with noise (same km region, different prices). That noise is why `J` will not reach 0.

### 11.2 Plot the regression line

On the **same** axes, draw the line \(y = \theta_0 + \theta_1 x\) from `xmin` to `xmax` of the dataset. After a correct train, the line should cut through the cloud with negative slope.

This is the fastest debug tool: if the line is horizontal at y = 0, you never trained or θ stayed 0. If the line is vertical or off-screen, scaling/α exploded.

### 11.3 Precision of the algorithm

“Precision” here means **how well the fitted line matches the known prices**, not “float64 vs float32.”

**MAE (mean absolute error)** — average error in **price units** (easy to say: “off by about N on average”):

\[
\text{MAE} = \frac{1}{m} \sum_i \lvert h_\theta(x^{(i)}) - y^{(i)} \rvert
\]

**MSE** — same as section 4; large mistakes dominate.

**RMSE** — \(\sqrt{\text{MSE}}\), same units as price.

**R² (coefficient of determination)** — “what fraction of the variance of price does our line explain?”

\[
R^2 = 1 - \frac{\sum_i (y^{(i)} - h_\theta(x^{(i)}))^2}{\sum_i (y^{(i)} - \bar{y})^2}
\]

where \(\bar{y}\) is the mean of prices.

- **R² = 1:** perfect fit.
- **R² = 0:** no better than always predicting the **average price**.
- **R² < 0:** worse than the average (broken training).

**Honest limitation for the oral:** we will compute these on the **same 24 points we trained on**. That measures **training fit**, not “accuracy on unseen cars.” The subject does not ask for a train/test split. You should still *say* that a real ML workflow would keep some cars out of training to test generalization.

Do **not** use a library that **fits** the model in order to report R². Computing the sums yourself is the point.

---

## 12. Data realities for *this* CSV

- **m = 24** — small. The line can still be reasonable, but one outlier moves θ more than it would on 10 000 cars.
- Prices are **not** a perfect line: two cars with similar km can differ by thousands. That is residual noise.
- **θ1 should be negative** after convergence.
- **θ0 should be positive** and larger than most prices (the intercept at 0 km is above the cloud if the slope is negative).
- Predicting mileage **outside** the dataset (e.g. 1 km or 1 000 000 km) is **extrapolation**. The formula still runs; the number may be nonsense (even **negative prices** if km is huge). You can mention clamping to 0 as product polish; the subject does not require it. Be ready to say: “linear model, no guarantee outside the observed km range.”

---

## 13. Edge cases (conceptually)

| Situation | Sensible behavior |
|-----------|-------------------|
| Train never run | Predict uses θ = 0 → price 0 |
| Empty CSV or m = 0 | Cannot compute 1/m; train must error clearly |
| Non-numeric CSV cell | Error; do not silently skip without a story |
| User types “abc” as mileage | Reject; ask again or exit with a message |
| Negative mileage | Physically odd; you may warn or still apply the formula — decide and defend |
| α too large | `J` explodes; retrace scaling and α |
| σ = 0 | All mileages equal; cannot standardize; data is degenerate |

---

## 14. End-to-end story (one paragraph to memorize)

We assume price is a **linear** function of mileage. We measure how wrong a candidate line is with **mean squared error** `J`. We start at θ = 0 and **subtract** α times the **average residual** (and residual × mileage) computed on **all m examples at once**, which is **batch gradient descent** and matches the subject’s tmp formulas. Because km and price have different scales, we **min-max scale mileage to [0, 1] while updating θ**, then **convert θ back** so predict stays `θ0 + θ1 * raw_mileage`. We save θ to a file. Before training, that file is absent and the estimate is 0. We judge success by a **decreasing `J`**, a **negative slope** through the scatter, and bonus metrics (MAE, R²) on the training set.

---

## 15. Evaluator-style questions (answers you should be able to say)

**What is m?**  
Number of training examples. Here, 24 rows in `data.csv`.

**What is the hypothesis?**  
`estimatePrice(mileage) = θ0 + θ1 * mileage`.

**Why is θ1 negative?**  
In the data, higher mileage tends to mean lower price. The slope of the best line is negative.

**What if I run predict before train?**  
θ0 = θ1 = 0, so the estimated price is 0.

**Why simultaneous update?**  
Both derivatives must be taken at the **same** current line. Updating θ0 first would change `estimatePrice` before tmpθ1 is finished; that is not the algorithm in the subject.

**The PDF writes tmpθ = α * (1/m) * Σ … — is tmp the new theta?**  
No. tmp is \(\alpha \times\) (partial derivative of `J`). New theta is `theta - tmp`.

**What is the learning rate?**  
α, the step size. Too large: diverge. Too small: slow. Chosen by watching `J`, not by the PDF.

**Did you use sklearn?**  
No. We implement the sums. Libraries do not fit the line.

**Why normalize?**  
Raw km makes tmpθ1 huge compared to tmpθ0, so GD diverges or needs an unusable α. We min-max scale x to [0, 1] for training, then denormalize θ so the saved formula still uses raw km.

**Is `J` the same as precision?**  
`J` / MSE is the quantity GD minimizes. “Precision” in the bonus is extra metrics (MAE, R², …) to *report* quality. R² is often nicer in an oral than raw `J`.

**Is this AI? Is this a neural network?**  
It is **supervised regression**, one linear neuron’s worth of math, **no hidden layers**. Neural nets use similar ideas (gradient descent) with many parameters and nonlinearities.

**Mitchell: T, E, P?**  
Task: predict price from km. Experience: the CSV. Performance: error metrics / `J`.

**Can the predicted price be negative?**  
Yes, if mileage is large enough that the line crosses below 0. Limitation of an unbounded linear model.

**Why not split train/test?**  
Not required. With 24 points, a split is noisy. We should still *know* that training error is optimistic.

---

## 16. Mini exercises (do these before asking to implement)

Do them on paper. If they are easy, Part A is landing.

1. For points `(0, 1)` and `(1, 3)`, write `h(x)` if θ0 = 1, θ1 = 2. Compute both residuals.  
2. With θ = 0, α = 1, m = 2, those two points: compute tmpθ0, tmpθ1, and the **new** θ after one simultaneous step.  
3. Explain in one sentence why summing residuals without squaring can look “perfect” when it is not.  
4. If all mileages are ~100 000 and you **do not** scale, which tmp (`θ0` or `θ1`) is typically much larger in magnitude, and why?  
5. Write the two lines of algebra that turn standardized θ' back into raw θ0, θ1.  
6. What does R² = 0 mean?  
7. Recite the update rule including the minus sign without looking.

---

# Part B — Implementation

We code in **small steps**. Each step has a goal, files touched, and a full explanation. Do not skip ahead: if you cannot explain step 1’s function without looking, we stay on step 1.

---

## B.0 Roadmap — files and steps

### Files we will have when the project is done

| File | Role |
|------|------|
| [dataset.py](dataset.py) | Read `data/data.csv` → lists of mileages, prices, and `m` |
| `hypothesis.py` | `estimate_price(mileage, theta0, theta1)` = the PDF formula |
| `thetas_io.py` | Save / load `θ0`, `θ1` (missing file → zeros) |
| `gradient_descent.py` | Cost `J`, PDF tmp sums, one step, **and** the full `fit` loop |
| `scaling.py` | Min-max mileage to `[0, 1]`, denormalize θ back to raw km |
| `train.py` | Load data, run **batch GD**, save thetas (mandatory program 2) |
| `predict.py` | Ask for mileage, print estimated price (mandatory program 1) |
| `plot.py` | Bonus: scatter of data + regression line |
| `precision.py` | Bonus: MAE, MSE, R² |
| [requirements.txt](requirements.txt) | Only `matplotlib` (plots). **No** sklearn / polyfit |
| [.gitignore](.gitignore) | Ignore venv, bytecode, generated `thetas.json` |
| `README.md` | How to run (last; subject-style usage) |
| `data/data.csv` | Given dataset |
| `thetas.json` | **Generated by train**, not written by hand |

Shared math lives in small modules so `train.py` / `predict.py` / `precision.py` do not copy-paste the hypothesis.

### Steps (we are on step 5)

| Step | What we build | Status |
|------|----------------|--------|
| **1** | Layout + `load_dataset` | done |
| **2** | `estimate_price` + save/load thetas | done |
| **3** | `predict.py` (prompt, θ=0 if untrained) | done |
| **4** | Cost `J` + one GD iteration (simultaneous, minus tmp) | done |
| **5** | Full training loop + **feature scaling and denormalize θ** | **in progress — stop here until understood** |
| **6** | `train.py` as the real program (α, iterations, write `thetas.json`) | not started |
| **7** | Bonus `plot.py` | not started |
| **8** | Bonus `precision.py` | not started |
| **9** | `README.md` + polish (bad input, empty CSV already partly in step 1) | not started |

---

## B.1 Step 1 — Load the dataset

### Goal

Turn the CSV into the three things the formulas need:

- `mileage[i]` → Python list `mileages`
- `price[i]` → Python list `prices`
- `m` → `len(mileages)`

No θ, no gradient descent yet. If this is wrong, every later sum is wrong.

### Files created in this step

- `.gitignore` — do not commit `.venv/`, `__pycache__/`, or `thetas.json` (thetas are **output** of train).
- `requirements.txt` — `matplotlib` for later plots. We do **not** install sklearn.
- `dataset.py` — the loader.

Optional on your machine (not committed as a folder):

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

matplotlib is unused until the plot step. Step 1 only needs the **standard library** (`csv`, `pathlib`).

### How to run it

From the project folder:

```text
python dataset.py
```

Expected idea of the output: `m = 24`, first row `km=240000, price=3650`, last row `km=61789, price=8290`, ranges matching Part A.

---

### `dataset.py` — line by line

**Imports**

- `csv` — parse comma-separated files correctly (quotes, headers). We do **not** `split(",")` by hand.
- `Path` from `pathlib` — file paths that work on Windows and Linux.

**`DEFAULT_CSV`**

```text
Path(__file__).resolve().parent / "data" / "data.csv"
```

- `__file__` is the path of `dataset.py`.
- `.parent` is the project folder, **not** the folder you happened to `cd` into.
- So `python dataset.py` still finds `data/data.csv` if you run it from elsewhere.

**`load_dataset(csv_path=DEFAULT_CSV)`**

1. **`Path(csv_path)` + `is_file()`**  
   If the file is missing, `FileNotFoundError`. Training cannot guess 24 cars out of thin air.

2. **Empty lists `mileages` and `prices`**  
   We will `append` once per valid row so index `i` is the same car in both lists. That is the subject’s `mileage[i]` and `price[i]`.

3. **`open(..., newline="", encoding="utf-8")`**  
   - `newline=""` is what the `csv` module documents: it handles line endings (`\n` vs `\r\n` on Windows).  
   - `utf-8` avoids locale surprises.

4. **`csv.DictReader`**  
   First row is the header. Each following row becomes a dict like `{"km": "240000", "price": "3650"}`. Keys are column names, not positions 0 and 1 — so if someone reorders columns later, we still work, as long as names stay `km` and `price`.

5. **Header check**  
   We require both names `km` and `price`. If the file is empty, `fieldnames` is `None` → error. If columns are `mileage,cost`, we fail **loudly**. Silent wrong columns would train on garbage.

6. **The row loop `enumerate(reader, start=2)`**  
   Line 1 is the header, so the first data row is line 2 in the file. That number is only for error messages.

7. **Skip a fully empty line**  
   Both cells blank → `continue`. A half-filled line is **not** skipped; it will fail numeric parse.

8. **`float(...)`**  
   CSV cells are strings. Gradient descent needs numbers. `float` accepts `3650` and `3650.0`. If the cell is `"abc"`, we raise `ValueError` with the line number. We do **not** drop bad rows quietly (that would change `m` without you knowing).

9. **`m = len(mileages)`**  
   This is the **m** in `1/m` in the PDF. If `m == 0`, `1/m` would crash or be meaningless later, so we error now.

10. **`return mileages, prices, m`**  
    Three values, order fixed. Callers unpack: `xs, ys, m = load_dataset()`.

**`if __name__ == "__main__":`**

This block runs **only** when you execute `python dataset.py`, not when `train.py` later does `from dataset import load_dataset`. It prints `m` and ranges so you can **see** the loader work without a debugger.

### Why this is not “the ML yet”

`load_dataset` does not learn. It only builds **experience E** in memory. `θ` stay nonexistent until train.

### Oral one-liners for this file

- “m is the number of rows I successfully parsed as numbers, here 24.”
- “I use the stdlib csv module; I do not fit a line in this file.”
- “km and price are parallel lists: same index, same car.”

### Check yourself before step 2

1. What does `m` represent, and why must it not be 0?  
2. Why two lists instead of a list of `(km, price)` pairs? (Pairs would also work; we chose parallel lists because they match `mileage[i]` / `price[i]` in the PDF.)  
3. What happens if `data.csv` is missing? If a row says `foo,1000`?  
4. Why `DictReader` instead of reading line 0 as data?  
5. Run `python dataset.py` and confirm `m = 24`.

When these are easy, say so and we do **step 2**: `estimate_price` and saving/loading θ.

---

## B.2 Step 2 — Hypothesis and persistence of θ

### Goal

Two independent pieces that train and predict will both import:

1. The **exact** subject formula, in one function, so we never write `θ0 + θ1 * x` in three different files with a typo.
2. A **file** that stores `θ0` and `θ1` between the two programs. Missing file = untrained = `(0, 0)`.

Still no gradient descent. Still no user prompt. We only make the hypothesis callable and θ serializable.

### Files created in this step

- [hypothesis.py](hypothesis.py) — `estimate_price`
- [thetas_io.py](thetas_io.py) — `save_thetas` / `load_thetas`

`thetas.json` is **not** in the repo. Train will create it later. `.gitignore` already ignores it.

### How to run it

```text
python hypothesis.py
python thetas_io.py
```

`hypothesis.py` should print `0.0` for an untrained estimate, and `7` for the toy line `1 + 2*3`.  
`thetas_io.py` should print `(0.0, 0.0)` from the default path (if you never trained), then round-trip a demo file and delete it.

---

### `hypothesis.py` — line by line

**`estimate_price(mileage, theta0, theta1)`**

```text
return theta0 + theta1 * mileage
```

This is:

\[
\text{estimatePrice}(\text{mileage}) = \theta_0 + \theta_1 \cdot \text{mileage}
\]

- **First argument is mileage**, not price. Mixing them would still “run” and give nonsense.
- **Multiplication before addition** is normal Python precedence: `θ0 + (θ1 * mileage)`, which is what we want. Parentheses are optional; we omit them to match the PDF.
- **`theta1 * mileage`**, never `theta1 * theta0`. θ0 is a price offset; it is not multiplied by km.
- Returns a **number** (float if any input is float). One car in, one estimated price out. The training loop will call this **m times** per iteration (once per example).

**Why a function at all?**  
Train and predict must use the **same** hypothesis. If predict used `θ0 * θ1 + mileage` by mistake, GD would minimize a different model than the one you demo. One function = one definition.

**`if __name__ == "__main__"`**

Two sanity checks from Part A:

- Untrained: `estimate_price(100000, 0, 0)` is **0**. Mileage does not matter when both θ are 0.
- Toy: `estimate_price(3, 1, 2)` is **7**, because `1 + 2*3 = 7`.

---

### `thetas_io.py` — line by line

**Why JSON?**  
A tiny object `{"theta0": ..., "theta1": ...}` is readable in a text editor during defense. You can open `thetas.json` after train and *see* a negative θ1. CSV would also work; JSON names the fields so we cannot swap columns silently.

**`DEFAULT_THETAS`**  
Same trick as the dataset: file next to the project (`thetas.json`), not “current working directory only”. Both `train.py` and `predict.py` will share this path by default.

**`save_thetas(theta0, theta1, path=DEFAULT_THETAS)`**

1. Wrap the path in `Path`.
2. Build a dict with **float** values so we never save a weird type.
3. `json.dumps(..., indent=2)` makes the file human-readable.
4. `write_text` creates or **overwrites** the file. Retraining replaces old θ.
5. Returns the path (useful for a print in train later).

This function does **not** train. It only writes two numbers it was given.

**`load_thetas(path=DEFAULT_THETAS)`**

1. If `path` is **not a file**: return `(0.0, 0.0)`.  
   That is the subject: *before running train, theta0 and theta1 are 0*. Predict must not crash; it must estimate 0.
2. If the file **exists**: `json.loads` the text, read keys `"theta0"` and `"theta1"`, convert with `float`.
3. If the file exists but is garbage (broken JSON, missing key, `"theta0": "hello"`): **raise**.  
   A corrupt file is not the same as “never trained”. Silent zeros would hide a bug after a “successful” train.

**Default arguments**  
`path=DEFAULT_THETAS` means callers can omit the path. Tests / the `__main__` demo pass another path so we do not pollute the real `thetas.json`.

**`if __name__ == "__main__"`**

- Load the **real** default path (zeros if untrained).
- Save `(1.5, -0.02)` to `_demo_thetas.json` (a throwaway name).
- Load that file back — you must see the same two numbers. That is **round-trip**: save then load is identity.
- `unlink()` deletes the demo file so the folder stays clean.

### How these two files connect later (not coded yet)

```text
train.py  →  compute θ  →  save_thetas(θ0, θ1)
predict.py →  load_thetas()  →  estimate_price(user_km, θ0, θ1)
```

The hypothesis does not know about files. The file helpers do not know about mileage. **Separation:** math vs disk.

### Oral one-liners for this step

- “The hypothesis is one line: theta0 plus theta1 times mileage.”
- “If thetas.json is missing, load returns 0 and 0, so the price is 0.”
- “We do not hardcode trained thetas in predict; we read them from the file train wrote.”

### Check yourself before step 3

1. Compute by hand: `estimate_price(10, 100, -2)`. What price should Python print?  
2. Why does `estimate_price(999999, 0, 0)` equal 0?  
3. Difference between “file missing” and “file present but invalid”?  
4. Why save JSON instead of pasting θ into `predict.py`?  
5. Run both `python hypothesis.py` and `python thetas_io.py`.

When these are easy, we do **step 3**: `predict.py` (prompt for mileage, load θ, print the estimate).

---

## B.3 Step 3 — The predict program

### Goal

Mandatory program 1 from the subject:

1. Launch the program.
2. It **prompts** for a mileage.
3. It prints the estimated price: `θ0 + θ1 * mileage`.
4. If train has **never** been run, `thetas.json` is missing, `load_thetas` returns `(0, 0)`, so the price is **0**.

This file does **not** read `data.csv` and does **not** update θ. Prediction is not learning.

### File created in this step

- [predict.py](predict.py)

It **imports** `estimate_price` and `load_thetas`. It does not copy the formula again.

### How to run it

From the project folder (no `thetas.json` yet):

```text
python predict.py
```

Type a mileage, for example `85000`. Expected estimate: **`0.0`** (untrained).

After we implement train (later steps), run train once, then predict again: the number should no longer be 0, and **higher km should give a lower price** (negative θ1).

### `predict.py` — line by line

**Imports**

- `estimate_price` — the PDF hypothesis.
- `load_thetas` — `(0, 0)` if the file is missing.

Predict does not import `dataset.py`. The CSV is **training** data. At predict time we only have a mileage from the **user**.

**`read_mileage()`**

```text
raw = input("Mileage (km): ").strip()
```

- `input(...)` prints the prompt and waits for a line from the keyboard. That is the subject’s “prompt you for a mileage.”
- `.strip()` removes spaces so `"  85000  "` still works.

Empty string → print a message and **loop again**.  
Non-numeric (`abc`) → `float(...)` raises `ValueError` → message and loop again.  
A valid number (`85000`, `85000.5`, even `0`) → `return float(raw)`.

**Why a loop?**  
The subject does not specify retries. One crash on `"abc"` would also be defensible. Retrying is nicer and still honest: we only accept a number.

**Negative mileage:** the formula is defined for any real `x`. This program **accepts** a negative number if the user types one. In the oral: physically odd; the linear model does not forbid it. We did not clamp.

**`main()`**

1. `theta0, theta1 = load_thetas()` — disk (or zeros).
2. `mileage = read_mileage()` — keyboard.
3. `price = estimate_price(mileage, theta0, theta1)` — hypothesis.
4. `print(f"Estimated price: {price}")` — show the result.

Order matters: we load θ **before** asking for km. θ do not depend on this one car.

**`if __name__ == "__main__": main()`**  
Running `python predict.py` starts `main`. If some later file imported `read_mileage`, it would not auto-prompt.

### What you should see (untrained)

| You type | Printed price | Why |
|----------|----------------|-----|
| `100000` | `0.0` | θ0 = θ1 = 0 |
| `0` | `0.0` | still 0 + 0×0 |
| `abc` | error message, ask again | not a float |

### Oral one-liners

- “Predict never trains. It only applies the saved line.”
- “No thetas.json means both thetas are 0, so the estimate is 0.”
- “The hypothesis is imported from hypothesis.py, same function train will use.”

### Check yourself before step 4

1. Run `python predict.py` **without** having trained. Confirm the estimate is 0.  
2. Why does predict not open `data.csv`?  
3. If you hardcoded `theta0 = 8000` inside `predict.py`, why would that violate the subject?  
4. Trace one call: user types `240000`, θ = 0, 0. Which function returns 0?  

When these are easy, we do **step 4**: cost `J` and **one** simultaneous gradient-descent update (still not the full train program).

---

## B.4 Step 4 — Cost `J` and one gradient step

### Goal

Translate Part A into Python **without** a training loop and **without** scaling:

1. Compute `J` for a given `(θ0, θ1)`.
2. Compute the subject’s `tmpθ0` and `tmpθ1` from **all m examples**, using the **same** current θ.
3. Apply `θ ← θ − tmp`.

After this step you can recode the PDF formulas from memory. Full train + min-max/standardize is **step 5**.

### File created in this step

- [gradient_descent.py](gradient_descent.py)

### How to run it

```text
python gradient_descent.py
```

**First block (toy):** must match Part A’s hand calculation.

| Quantity | Expected |
|----------|----------|
| tmp0 | `-0.4` |
| tmp1 | `-0.65` |
| new θ0 | `0.4` |
| new θ1 | `0.65` |
| `J` after | **smaller** than `J` before |

**Second block (`data.csv`, α = 0.1, raw km):** θ1 becomes huge and `J` usually **increases**. That is not a bug in the sums; it is the scale problem. Step 5 exists because of this.

---

### `cost_j` — line by line

\[
J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=0}^{m-1} \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2
\]

- Loop `i` from `0` to `m-1`.
- `error = estimate_price(...) - prices[i]` — **estimate minus real**, same order as the PDF.
- `total += error * error` — square; do not add `error` alone.
- Divide by `2 * m`, not only `m`. The extra `2` cancels in the derivative and yields the PDF’s `(1/m) Σ error`.

`J` is a **score for the current line**. It does not change θ.

---

### `compute_tmps` — the PDF, nothing else

\[
\text{tmp}\theta_0 = \alpha \cdot \frac{1}{m} \sum_i (\text{estimate} - \text{price})
\]
\[
\text{tmp}\theta_1 = \alpha \cdot \frac{1}{m} \sum_i (\text{estimate} - \text{price}) \cdot \text{mileage}
\]

**Simultaneous:** the loop only **reads** `theta0` and `theta1`. It never assigns to them. Both sums see the same line.

Two running totals:

- `sum_error` → for tmpθ0 (no extra `× mileage`)
- `sum_error_x` → for tmpθ1 (`error * mileages[i]`)

Then multiply by `learning_rate * (1 / m)`.

**tmp is not the new θ.** If you `return tmp0, tmp1` and use them as θ, you are not doing gradient descent.

---

### `gradient_step` — the minus sign

```text
tmp0, tmp1 = compute_tmps(...)
return theta0 - tmp0, theta1 - tmp1
```

Both subtractions use tmps that were computed **before** either θ changed. That is simultaneous update.

If tmp is negative (underestimates, like θ = 0 on car prices), **minus a negative increases θ**. That matches Part A.

---

### Why the toy is in `__main__`

The two points `(1, 3)` and `(2, 5)`, α = 0.1, start at 0, are the **same numbers** as in Part A section 5. If Python disagrees with your paper, the code is wrong (sign, `1/m`, or updating θ inside the loop).

The CSV block is a **demo of failure without scaling**, not a trained model. Do not save those huge θ into `thetas.json`.

### Oral one-liners

- “tmp is alpha times the average residual (and residual times x). New theta is old theta minus tmp.”
- “I compute both tmps, then I apply both updates. I never use the new θ0 inside tmpθ1.”
- “m is the length of the lists; every iteration uses every example (batch).”
- “On raw km, one step with α = 0.1 blows up θ1. We will scale mileage next.”

### Check yourself before step 5

1. Run `python gradient_descent.py`. Confirm the toy prints `-0.4`, `-0.65`, `0.4`, `0.65`.  
2. If `J` after the toy step is **larger** than before, which sign did you likely flip?  
3. Why are there two accumulators in `compute_tmps`?  
4. Why does the CSV one-step print a gigantic θ1?  
5. Recite: `θ := θ − tmp`, tmp from the PDF, same θ for both tmps.

When these are easy, we do **step 5**: loop many iterations **and** scale mileage for GD, then convert θ back to raw km.

---

## B.5 Step 5 — Training loop, min-max `[0, 1]`, denormalize θ

### Goal

1. Repeat `gradient_step` many times (the **training loop**).
2. Do those steps on **scaled** mileage so α can be a normal number like `0.5`.
3. Convert the fitted line back to **raw km** so `predict.py` still uses the PDF formula.
4. **Do not** write `thetas.json` yet — that is `train.py` in step 6.

**Decision (oral):** we scale with **min-max into `[0, 1]`**, not z-score. Same idea as Part A section 7; this is the variant we coded.

### Files in this step

- [scaling.py](scaling.py) — bounds, scale list, denormalize θ
- [gradient_descent.py](gradient_descent.py) — new function `fit_linear_regression`

### How to run it

```text
python gradient_descent.py
```

You still see the **toy** (step 4) and the **exploding** one-step on raw km. Then a third block:

- `J start` large
- `J iter 1000`, `2000`, … **decreasing**
- `J end` on **raw** x with denormalized θ — should match the last scaled `J` (same predictions)
- `theta1` **negative**
- estimate at 50 000 km **higher** than at 200 000 km

Default hyperparameters (you choose them; they are not in the PDF): `learning_rate=0.5`, `n_iterations=10000`.

---

### `scaling.py` — line by line

**`min_max_bounds(mileages)`**  
`xmin = min(...)`, `xmax = max(...)`. If they are equal, every car has the same km: the span is 0, division is impossible → error.

**`scale_to_unit_interval`**

\[
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]

- The smallest km in the CSV becomes `0`.
- The largest becomes `1`.
- Everything else is in between.

We scale **only mileage**. Price stays in real units (euros / whatever the CSV is).

**`denormalize_thetas`**

GD found a line in `x'` space: \( h = \theta_0' + \theta_1' x' \).

Substitute \( x' = (x - x_{\min}) / \text{span} \):

\[
h = \theta_0' + \theta_1'\frac{x - x_{\min}}{\text{span}}
= \left(\theta_0' - \theta_1'\frac{x_{\min}}{\text{span}}\right)
+ \left(\frac{\theta_1'}{\text{span}}\right) x
\]

So:

- `theta1 = theta1_scaled / span`
- `theta0 = theta0_scaled - theta1_scaled * xmin / span`

**We do not save xmin/xmax for predict.** They are folded into θ. After this, `estimate_price(raw_km, theta0, theta1)` is the same number as `estimate_price(scaled_km, theta0_scaled, theta1_scaled)`.

---

### `fit_linear_regression` — line by line

1. **Bounds + scaled copy of mileages.** Original `mileages` list is unchanged (we still need it to check `J` in raw space).
2. **Start `theta0_s = theta1_s = 0`** — same as the subject, but these zeros are in **scaled** space.
3. **Loop `n_iterations` times:**  
   `theta0_s, theta1_s = gradient_step(scaled, prices, ...)`.  
   That is still the **PDF** update. The only trick is: the `mileage[i]` inside tmpθ1 is **x'**, not raw km.
4. **Print `J` every `log_every` steps** on scaled x (cheap sanity check: must go down).
5. **Denormalize** → return raw `theta0`, `theta1`.
6. Print `J` again using **raw** mileages and raw θ. If denormalize is correct, this `J` equals the last scaled `J`.

**Why not scale inside `compute_tmps`?**  
Keep the PDF function stupid: it always uses the x it is given. Scaling is a **preprocessing** choice around that function, then an **algebra** fix after.

### What we still do not do

- No `save_thetas` (step 6).
- No plot (step 7).
- Prices are not scaled.
- No sklearn.

### Oral one-liners

- “I min-max km to 0–1 so tmpθ1 is not a billion times tmpθ0.”
- “Gradient descent is still the subject’s sums; x in the sum is scaled km.”
- “I rewrite θ so predict uses raw km and never sees xmin/xmax.”
- “θ1 after denormalize is negative: more km, lower price.”

### Check yourself before step 6

1. Run `python gradient_descent.py`. Compare exploding one-step vs the scaled loop.  
2. Why is the smallest mileage mapped to 0 and the largest to 1?  
3. Write the two formulas for raw θ0 and θ1 from θ', xmin, xmax.  
4. Why can predict stay `θ0 + θ1 * mileage` with no scaler object?  
5. If `J` on raw x after denormalize **disagrees** with last scaled `J`, what broke?

When these are easy, we do **step 6**: `train.py` — load CSV, call `fit_linear_regression`, **save** `thetas.json`.

---

## B.6 Step 6 — `train.py` (mandatory program 2)

### Goal

Wire every piece together into the program the subject calls "the training program":

1. Load the dataset.
2. Call `fit_linear_regression` (which does GD + scaling inside).
3. Report θ and a sanity check on the sign of θ1.
4. **Write `thetas.json`** — the only output of training.

`train.py` itself has **zero math**. All math lives in `gradient_descent.py` and `scaling.py`. This file is the orchestrator.

### File created in this step

- [train.py](train.py)

### How to run it

```text
python train.py
python train.py --lr 0.1 --iters 5000
```

Expected output (values depend on α and iterations but sign must match):

```
theta0 =  8499.599650
theta1 =    -0.021449
theta1 is negative -- correct (more km = lower price).
Thetas saved to: .../thetas.json
```

---

### `train.py` — line by line

**Imports**

- `argparse` — standard library; lets the user pass `--lr` and `--iters` on the command line without editing the file.
- `load_dataset` — experience E.
- `cost_j`, `fit_linear_regression` — the core GD machinery.
- `save_thetas` — puts θ on disk.

**`parse_args()`**

```text
--lr    float   default 0.5   learning rate (α)
--iters int     default 10000 number of GD iterations
```

These are **hyperparameters**: you chose them; gradient descent does not learn them. `argparse` makes it easy to try different values at the command line without touching the source.

**`main()` — four sections**

1. **Load** — `load_dataset()` and print `m`, km range, price range. Just reporting what experience E is. No θ yet.

2. **Train** — one call to `fit_linear_regression(mileages, prices, learning_rate, n_iterations)`.
   - Inside that function: min-max scale mileage, run the loop, denormalize.
   - `train.py` does **not** know about xmin, xmax, or scaled θ. Those details stay in `gradient_descent.py` and `scaling.py`.
   - Returns raw `theta0`, `theta1` already in the unit of the PDF formula.

3. **Report** — print both θ. The `if theta1 > 0` guard is a sanity check, not business logic. If the slope comes out positive after training, something is wrong (sign flip, no scaling, α too large). Printing a WARNING now saves you from a silent wrong model.

4. **Save** — `save_thetas(theta0, theta1)` writes `thetas.json`. From this moment, `predict.py` will give a real estimate instead of 0.

**`if __name__ == "__main__": main()`**

Same pattern as every other file. `python train.py` runs the program; `from train import ...` (if someone wanted to) would not auto-execute.

### Why train.py has no math

The subject's update formula is in `compute_tmps` / `gradient_step`. Scaling is in `scaling.py`. Persistence is in `thetas_io.py`. `train.py` only calls them in the right order. This separation means:

- You can unit-test `cost_j` without running the whole program.
- `precision.py` and `plot.py` can reuse `load_dataset` and `load_thetas` without re-implementing them.
- An evaluator can point at any one file and ask "what does this do?" — each file has one job.

### What changes in `thetas.json` after training

Before: file is absent (or all zeros).  
After:

```json
{
  "theta0": 8499.5996...,
  "theta1": -0.021448...
}
```

θ1 is negative — the model has learned that higher mileage → lower price.

### Oral one-liners for this step

- "train.py has no math. It loads, calls fit_linear_regression, and saves. The math is in gradient_descent.py."
- "α and iterations are hyperparameters — I picked them by watching J decrease, not by a formula."
- "After train.py writes thetas.json, predict.py stops returning 0."
- "The theta1 sign check is a guard: positive slope after training means something broke."

### Check yourself before step 7

1. Run `python train.py`. Open `thetas.json`. Is θ1 negative?
2. Run `python train.py --lr 10`. What happens to J? Why?
3. Why does `train.py` call `save_thetas` instead of `open("thetas.json", "w")` directly?
4. If you ran `train.py` twice in a row, would the second run overwrite the first `thetas.json`? Is that correct?
5. Trace the call chain: `train.py` → `fit_linear_regression` → `gradient_step` → `compute_tmps` → `estimate_price`. Name the file each function lives in.

When these are easy, we do **step 7**: `plot.py` — scatter of the 24 cars with the regression line on top.

---

## B.7 Step 7 — `plot.py` (bonus: scatter + regression line)

### Goal

Draw the two things the subject calls "plot the data" and "plot the line":

1. Each car as a **dot** `(km, price)` — the scatter, i.e. experience E.
2. The **regression line** `h(x) = θ0 + θ1 × x` from xmin to xmax — the model.

This is the fastest debug tool in the project. If the line is horizontal at y = 0, training never ran. If it is off the screen, α exploded. If the slope is positive, scaling broke.

### File created in this step

- [plot.py](plot.py)

`plot.py` does **not** train. It only reads what `train.py` already saved.

### How to run it

```text
python train.py    # must come first
python plot.py
```

A window opens with the scatter and line. The file `plot.png` is also written so the evaluator can see the result without running the code.

---

### `plot.py` — line by line

**Imports**

- `matplotlib.pyplot` — the only library in `requirements.txt`. Allowed: it draws, it does not fit.
- `load_dataset` — the 24 `(km, price)` pairs for the scatter.
- `estimate_price` — called to compute y values on the line. Same function as `predict.py`.
- `load_thetas` — reads the trained θ from disk.

**`plot_regression(mileages, prices, theta0, theta1)`**

Three visual layers on the same axes:

**Layer 1 — Scatter (the data)**

```python
ax.scatter(mileages, prices, ...)
```

Each call to `scatter` places one dot per car. The horizontal axis is km; the vertical axis is price. You should see a **downward cloud**: more km, lower price, with visible noise (cars at the same km but different prices).

**Layer 2 — Regression line (the model)**

```python
x_line = [min(mileages), max(mileages)]
y_line = [estimate_price(x, theta0, theta1) for x in x_line]
ax.plot(x_line, y_line, ...)
```

A straight line needs only **two points**. We use the leftmost and rightmost km in the dataset. The y values are computed with `estimate_price` — the **exact same function** `predict.py` uses. What you see is literally the line that the saved θ represent.

Label on the legend: `h(x) = 8499 + -0.02145 * km`. The evaluator can read the formula directly off the plot.

**Layer 3 — Prediction crosshair (annotation)**

```python
example_km = 100_000
example_price = estimate_price(example_km, theta0, theta1)
ax.axvline(x=example_km, ...)   # vertical dashed line at 100 000 km
ax.axhline(y=example_price, ...) # horizontal dashed line at estimated price
```

This is a visual demo of one prediction: "for 100 000 km, the model says ~6 355." Connects the abstract formula to a real answer.

**`plt.savefig("plot.png", ...)`**

Writes the image to disk before `plt.show()`. Even if the evaluator closes the window immediately, `plot.png` persists.

**`main()`**

1. `load_dataset()` — 24 points.
2. `load_thetas()` — the trained line.
3. Guard: if both θ are 0, print a warning. The scatter still appears, but the "line" is a flat y = 0, which is visually wrong and tells you to train first.
4. Call `plot_regression(...)`.

### What the plot tells you at a glance

| What you see | Diagnosis |
|---|---|
| Downward cloud of dots, red line cutting through with negative slope | Correct ✓ |
| Line flat at y = 0 | Never trained (θ = 0) |
| Line way off screen or vertical | α too large, GD diverged |
| Line slopes upward (positive slope) | θ1 > 0 — scaling or sign error |
| Dots and line don't overlap at all | Wrong CSV column or sign flip |

### Oral one-liners for this step

- "The scatter shows the 24 training points. The line is the result of gradient descent — the same formula predict.py uses."
- "I only need two x values to draw a straight line: xmin and xmax of the dataset."
- "I call estimate_price to get the y values — it's the same function the user calls at predict time."
- "The crosshair shows an example prediction at 100 000 km so the evaluator can see a concrete answer."

### Check yourself before step 8

1. Run `python plot.py`. Is the slope negative? Does the line cut through the cloud?
2. Why do we only pass `[min(mileages), max(mileages)]` to `ax.plot` and not all 24 km values?
3. If the line were at y = 0, what would you do to fix it?
4. Why is `plot.py` not inside `train.py`?
5. What does `plot.png` allow the evaluator to do that `plt.show()` alone does not?

When these are easy, we do **step 8**: `precision.py` — MAE, MSE, RMSE, R².

---

## B.8 Step 8 — `precision.py` (bonus: quality metrics)

### Goal

Report **how well** the fitted line matches the 24 training points using four metrics:

- **MAE** — average absolute miss in price units. Most human-readable.
- **MSE** — average squared error. What GD was minimizing (divided by m, not 2m).
- **RMSE** — square root of MSE; same units as price.
- **R²** — fraction of price variance explained by the line. The one evaluators ask about most.

All sums are written as **plain Python loops**. No library computes the fit. Using `sklearn.metrics.r2_score` here would still be cheating: the rule is "no library that does the ML work." Computing R² manually from `estimate_price` is fine.

### File created in this step

- [precision.py](precision.py)

### How to run it

```text
python train.py      # first
python precision.py
```

Expected output (our dataset):

```
MAE  : 557.84
MSE  : 445645.25
RMSE : 667.57
R²   : 0.7330   (line explains 73 % of price variance)
```

---

### `precision.py` — line by line

**`mean_absolute_error`**

\[
\text{MAE} = \frac{1}{m} \sum_i \bigl| h_\theta(x^{(i)}) - y^{(i)} \bigr|
\]

- `abs(estimate_price(...) - prices[i])` — the vertical distance between the line and each dot.
- Every error counts equally regardless of size (no squaring).
- Defense sentence: "On average we miss by about 558 price units."

Note the order: `estimate − price`, same as the residual in Part A. The absolute value makes it symmetric — it does not matter which way.

**`mean_squared_error`**

\[
\text{MSE} = \frac{1}{m} \sum_i \bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2
\]

- Same sum as `cost_j` in `gradient_descent.py`, but divided by **m**, not **2m**.
- The `1/2` in `cost_j` was only there so the derivative matched the PDF. For reporting quality, MSE with `1/m` is the standard definition.
- Large errors dominate: a miss of 1 000 contributes `1 000 000` to the sum, while a miss of 10 contributes only `100`.

**`root_mean_squared_error`**

```python
return math.sqrt(mean_squared_error(...))
```

Puts MSE back in **price units**. Because squaring exaggerates large errors, RMSE > MAE almost always. Here: RMSE 668 > MAE 558. The gap tells you there are a few cars where the line is more wrong than average.

**`r_squared`**

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

- **SS\_res** (residual sum of squares): how much error remains after fitting the line.
  - `ss_res += (prices[i] - estimate_price(...)) ** 2`
- **SS\_tot** (total sum of squares): how spread out prices are around their own mean.
  - `mean_price = sum(prices) / m`
  - `ss_tot += (prices[i] - mean_price) ** 2`

Intuition: if `SS_res = 0`, the line is perfect → R² = 1. If `SS_res = SS_tot`, the line is no better than "always predict the mean price" → R² = 0. If `SS_res > SS_tot`, the line is **worse** than predicting the mean → R² < 0 (broken training).

**Our result: R² = 0.733** — the line explains 73% of why car prices vary. The remaining 27% is noise that a single straight line cannot capture (two cars at the same km but very different prices).

**Edge case: `ss_tot == 0`**

If all 24 prices are identical, the denominator is 0. This cannot happen with `data.csv`, but we guard it anyway: in that degenerate case, any line is trivially "perfect," so we return 1.0.

**`main()`**

1. Load data and θ.
2. Guard: if θ = 0, warn (untrained).
3. Call all four metric functions. Each does its own loop — no state shared between them.
4. Print results with interpretation strings.
5. Print the honest disclaimer: all metrics are on the **training set**, which is optimistic.

### Relationship between J and MSE

| | Formula | Used for |
|---|---|---|
| `cost_j` | `(1/2m) Σ error²` | GD (the `1/2` cancels in the derivative) |
| MSE | `(1/m) Σ error²` | Reporting quality |

They have the **same minimum** (same θ). `cost_j = MSE / 2`. Both measure the same thing; the constant factor just makes calculus cleaner inside GD.

### Oral one-liners for this step

- "MAE is the average miss in price units — the easiest number to say at defense."
- "R² = 0.73 means our line explains 73% of the variance in prices. The other 27% is noise a straight line can't capture."
- "If R² were negative, the line is worse than just predicting the average price — that would mean training broke."
- "All four metrics are computed on the training set, which is optimistic. We don't have a separate test set — the subject doesn't require one."
- "MSE and cost J have the same minimum. The 1/2 in J is just to simplify the derivative."

### Check yourself before step 9

1. R² = 0 — what does that mean in plain words?
2. Why is RMSE always ≥ MAE? (Hint: squaring then rooting amplifies large errors.)
3. Why do we not use `sklearn.metrics.r2_score` even though it exists?
4. If you deleted `thetas.json` and ran `python precision.py`, what would the output be?
5. Compute by hand: for one car, estimate = 7 000, price = 6 500. What is its contribution to MAE? To MSE?

When these are easy, we do **step 9**: `README.md` and final polish.

---

## B.9 Step 9 — `README.md` and final polish

### Goal

The last deliverable before pushing:

1. A `README.md` that lets a stranger (or evaluator) clone the repo and run both mandatory programs in under two minutes.
2. Confirm all edge cases that `LEARNING.md` section 13 lists are actually handled in code.
3. Fill in the **AI use** section of this file.

### File created in this step

- [README.md](README.md)

### What README.md must cover (subject expectation)

The subject asks "how to use" both programs. At minimum:

```text
python train.py
python predict.py
```

Our README goes further:

| Section | Why it matters |
|---|---|
| One-paragraph algorithm summary | Evaluator can read it before opening any `.py` file |
| Setup (venv + pip) | Grader can reproduce the environment |
| Usage for all 4 scripts | Including optional `--lr` and `--iters` flags |
| Module map table | Shows you know which file does what |
| Edge cases table | Proves robustness was considered |
| Algorithm block | Formulas in ASCII for quick review |
| AI use | Required by 42 honesty policy |

### Edge cases — where they live in code

Go through LEARNING.md section 13 and match each row to a file:

| Situation | File | What happens |
|---|---|---|
| Train never run | `thetas_io.py` `load_thetas` | Returns `(0.0, 0.0)` → predict prints 0 |
| Empty CSV / m = 0 | `dataset.py` | `ValueError: No training examples` |
| Non-numeric CSV cell | `dataset.py` | `ValueError` with line number |
| User types "abc" as mileage | `predict.py` `read_mileage` | Caught by `except ValueError`; loops again |
| Empty mileage input | `predict.py` `read_mileage` | `if raw == ""` branch; loops again |
| α too large | `train.py` + output | J printed per 1000 iterations — you see it explode |
| σ = 0 (all km equal) | `scaling.py` `min_max_bounds` | `ValueError: All mileages are equal` |
| Negative mileage | `predict.py` | Accepted — formula is defined for any real x; oral: "linear model, no clamp" |

Every one of these was already in the code from earlier steps. Step 9 is about **knowing where they are**, not adding new code.

### Oral one-liners for this step

- "The README shows how to set up the venv, run train, and run predict. It has nothing extra."
- "Each edge case is handled in the file closest to where the error occurs — the loader checks the CSV, the prompt checks the user input."
- "The AI use section lists what AI explained and what I decided. I can defend every decision without the README."

### Check yourself — final

1. Clone your own repo fresh. Follow only the README. Does `python train.py` then `python predict.py` work?
2. Delete `thetas.json`. Run `python predict.py`. Does it print 0 without crashing?
3. Run `python train.py --lr 50`. Watch J. Can you explain what you see?
4. Open `thetas.json`. Is θ1 negative? What would it mean if it were not?
5. Recite from memory (no file open): the hypothesis formula, the two update formulas including the minus, and why we scale.

If all five are easy, the project is done.

---

## AI use

This project was built step by step with AI assistance (Antigravity / Gemini) for:

- **Explanations**: every concept in Part A was discussed interactively until it could be reconstructed from memory.
- **Code review**: each function was checked against the Part A formulas and the hand-calculated toy examples before being kept.
- **Catching edge cases**: the guard for `θ1 > 0`, the `ss_tot == 0` case in R², and the Windows `UnicodeEncodeError` on the `θ` character were found during live test runs.

Technical decisions made **by the student** (not delegated to AI):

- Choice of **min-max** scaling over z-score (simpler algebra for denormalization, easier to explain at defense).
- Default `α = 0.5`, `n_iterations = 10 000` (chosen by watching J flatten, not by a formula).
- **Not** clamping negative prices (the linear model is unbounded; we disclose this at defense rather than hiding it).
- File layout: one function per concept, no copy-pasted formulas across files.
- **Plain Python loops** for all sums instead of numpy, to match the PDF's Σ notation exactly.

Every formula — the hypothesis, the tmp sums, the simultaneous update, the denormalization algebra, MAE, MSE, R² — can be reconstructed from memory and verified against the hand calculations in Part A of this file.
