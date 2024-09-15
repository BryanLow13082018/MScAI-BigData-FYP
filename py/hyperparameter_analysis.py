import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from optuna.visualization import plot_param_importances, plot_optimization_history

def plot_hyperparameter_importance(study):
    """
    Plot the importance of each hyperparameter.
    
    Args:
        study (optuna.study.Study): The completed Optuna study object.
    
    Returns:
        plotly.graph_objs._figure.Figure: A plotly figure object showing hyperparameter importance.
    """
    fig = plot_param_importances(study)
    fig.update_layout(title="Hyperparameter Importance")
    return fig

def plot_optimization_history(study):
    """
    Plot the optimization history.
    
    Args:
        study (optuna.study.Study): The completed Optuna study object.
    
    Returns:
        plotly.graph_objs._figure.Figure: A plotly figure object showing optimization history.
    """
    fig = plot_optimization_history(study)
    fig.update_layout(title="Optimization History")
    return fig

def plot_parallel_coordinate(study):
    """
    Create a parallel coordinate plot for hyperparameter visualization.
    
    Args:
        study (optuna.study.Study): The completed Optuna study object.
    
    Returns:
        plotly.graph_objs._figure.Figure: A plotly figure object showing parallel coordinates plot.
    """
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = study.best_trial.values,
                        colorscale = 'Viridis',
                        showscale = True),
            dimensions = [
                dict(range = [study.best_trials[0].params[param] for param in study.best_trials[0].params],
                     label = param,
                     values = [trial.params[param] for trial in study.best_trials])
                for param in study.best_trials[0].params
            ]
        )
    )
    fig.update_layout(title="Parallel Coordinate Plot of Hyperparameters")
    return fig

def analyze_hyperparameter_sensitivity(study):
    """
    Analyze the sensitivity of each hyperparameter.
    
    Args:
        study (optuna.study.Study): The completed Optuna study object.
    
    Returns:
        list of tuple: Sorted list of (parameter, sensitivity) tuples.
    """
    param_names = list(study.best_params.keys())
    sensitivity = {}
    
    for param in param_names:
        values = [trial.params[param] for trial in study.trials]
        objective_values = [trial.value for trial in study.trials]
        
        # Calculate correlation between parameter values and objective values
        correlation = np.corrcoef(values, objective_values)[0, 1]
        sensitivity[param] = abs(correlation)
    
    # Sort sensitivities
    sorted_sensitivity = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_sensitivity

def plot_sensitivity_analysis(sensitivity):
    """
    Plot the results of sensitivity analysis.
    
    Args:
        sensitivity (list of tuple): Sorted list of (parameter, sensitivity) tuples.
    
    Returns:
        plotly.graph_objs._figure.Figure: A plotly figure object showing sensitivity analysis.
    """
    params, sensitivities = zip(*sensitivity)
    fig = px.bar(x=params, y=sensitivities)
    fig.update_layout(
        title="Hyperparameter Sensitivity Analysis",
        xaxis_title="Hyperparameter",
        yaxis_title="Sensitivity (Absolute Correlation)"
    )
    return fig