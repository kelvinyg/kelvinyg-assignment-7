from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # TODO 1: Generate a random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)  # Replace with code to generate random values for X

    # TODO 2: Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Replace with code to generate Y

    # TODO 3: Fit a linear regression model to X and Y
    model = LinearRegression()  # Initialize the LinearRegression model
    # None  # Fit the model to X and Y
    X_reshaped = X.reshape(-1, 1)  # Reshape X for model fitting
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]   # Extract the slope (coefficient) from the fitted model
    intercept = model.intercept_ # Extract the intercept from the fitted model

    # 4. Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color='blue', alpha=0.5, label='Data Points')
    X_line = np.linspace(0, 1, 100)
    Y_line = slope * X_line + intercept
    plt.plot(X_line, Y_line, color='red', label=f'Regression Line: Y = {slope:.2f}X + {intercept:.2f}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with Fitted Regression Line")
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # TODO 5: Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []

    for _ in range(S):
        # TODO 6: Generate simulated datasets using the same beta0 and beta1
        X_sim = np.random.uniform(0, 1, N)  # Replace with code to generate simulated X values
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)  # Replace with code to generate simulated Y values

        # TODO 7: Fit linear regression to simulated data and store slope and intercept
        sim_model = LinearRegression()  # Replace with code to fit the model
        X_sim_reshaped = X_sim.reshape(-1, 1)
        sim_model.fit(X_sim_reshaped, Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # TODO 8: Plot histograms of slopes and intercepts

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Replace with code to generate and save the histogram plot

    # TODO 9: Return data needed for further analysis, including slopes and intercepts
    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = sum(s > slope for s in slopes) / S if S > 0 else 0
    intercept_extreme = sum(i < intercept for i in intercepts) / S if S > 0 else 0
    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store data in session
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    p_value = None
    fun_message = ""
    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # TODO 10: Calculate p-value based on test type
    if test_type == 'greater':
        p_value = np.sum(simulated_stats >= observed_stat) / S
    elif test_type == 'less':
        p_value = np.sum(simulated_stats <= observed_stat) / S
    elif test_type == 'not_equal':
        extreme_value = np.abs(observed_stat - hypothesized_value)
        p_value = np.sum(np.abs(simulated_stats - hypothesized_value) >= extreme_value) / S

    # TODO 11: If p_value is very small (e.g., <= 0.0001), set fun_message to a fun message
    if p_value is not None and p_value <= 0.0001:
        fun_message = "Whoa! You've encountered a rare event!"

    # TODO 12: Plot histogram of simulated statistics
    # Replace with code to generate and save the plot
    plt.figure(figsize=(8, 6))
    plt.hist(simulated_stats, bins=20, alpha=0.5, label='Simulated Values')
    plt.axvline(observed_stat, color='red', linestyle='--', label=f"Observed {parameter.capitalize()}")
    plt.axvline(hypothesized_value, color='green', linestyle='--', label=f"Hypothesized {parameter.capitalize()}")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100  # Convert to decimal

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = float(session.get("slope"))
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = float(session.get("intercept"))
        true_param = beta0

    # Check that S > 1
    if S <= 1:
        error_message = "Number of simulations (S) must be greater than 1 to calculate a confidence interval."
        return render_template("index.html", error_message=error_message)

    # Calculate mean and standard error of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)  # Sample standard deviation
    standard_error = std_estimate / np.sqrt(S)

    # Check for zero standard error
    if standard_error == 0:
        error_message = "Standard error is zero; cannot compute confidence interval."
        return render_template("index.html", error_message=error_message)

    # Calculate the t-value for the given confidence level
    t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, df=S - 1)
    margin_of_error = t_value * standard_error
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot the individual estimates as gray points and confidence interval
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(estimates)), estimates, color='gray', alpha=0.5, label='Simulated Estimates')
    plt.axhline(mean_estimate, color='blue', linestyle='-', label=f'Mean Estimate = {mean_estimate:.4f}')
    plt.axhline(ci_lower, color='green', linestyle='--', label=f'CI Lower = {ci_lower:.4f}')
    plt.axhline(ci_upper, color='green', linestyle='--', label=f'CI Upper = {ci_upper:.4f}')

    # Mark the true parameter value on the plot
    if includes_true:
        plt.axhline(true_param, color='purple', linestyle='-', label=f'True {parameter.capitalize()} = {true_param:.4f} (Included)')
    else:
        plt.axhline(true_param, color='red', linestyle='-', label=f'True {parameter.capitalize()} = {true_param:.4f} (Excluded)')

    plt.title(f'{int(confidence_level * 100)}% Confidence Interval for {parameter.capitalize()}')
    plt.xlabel('Simulation Index')
    plt.ylabel('Estimate Value')
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=int(confidence_level * 100),  # Convert back to percentage
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
