import inspect
import numpy as np
import math
import random
import copy
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .models import Model
from .optimizers import OPTIMIZERS
from .functions import SCORING_FUNCTIONS


def __filter_optimizer_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter out invalid optimizer parameters from the parameter combination.

    Parameters
    ----------
    params : dict
        Dictionary containing hyperparameter keys and values. If it includes "optimizer" and "opt_params",
        the corresponding optimizer class is checked for valid parameters.

    Returns
    -------
    dict
        Dictionary with invalid optimizer parameters removed.
    """
    if "optimizer" in params and "opt_params" in params:
        optimizer_class = OPTIMIZERS.get(params["optimizer"], None)
        if optimizer_class is not None:
            # Get valid parameters from the optimizer's __init__ (excluding "self")
            valid_params = [
                key for key in inspect.signature(optimizer_class.__init__).parameters
                if key != "self"
            ]
            params["opt_params"] = {
                key: value for key, value in params["opt_params"].items()
                if key in valid_params
            }
    return params


def __extract_model_init_params(model_class: Type[Model], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract valid constructor parameters for the given model class from a parameter dictionary.

    Parameters
    ----------
    model_class : Model
        The model class to be instantiated.
    params : dict
        Dictionary containing potential parameters for the model.

    Returns
    -------
    dict
        Filtered dictionary of parameters matching the model_class constructor signature.
    """
    constructor_params = inspect.signature(model_class.__init__).parameters
    init_params = {}
    for key, _ in constructor_params.items():
        if key == "self":
            continue
        if key in params:
            init_params[key] = params[key]
    return init_params


def __create_folds(n_samples: int, cv: int) -> List[Tuple[int, int]]:
    """
    Create fold indices for cross-validation.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    cv : int
        Number of folds.

    Returns
    -------
    list of tuple
        List of (start_index, end_index) tuples for each fold.
    """
    fold_size = math.ceil(n_samples / cv)
    folds = []
    start = 0
    for _ in range(cv):
        end = min(start + fold_size, n_samples)
        folds.append((start, end))
        start = end
    return folds


def __generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generate all combinations of parameters from the parameter grid.

    Parameters
    ----------
    param_grid : dict
        Dictionary where keys are parameter names and values are lists of possible parameter values.

    Returns
    -------
    list of dict
        List of dictionaries, each representing a unique combination of parameters.
    """
    # Ensure that "opt_params" exists in the grid
    if "opt_params" not in param_grid or not param_grid["opt_params"]:
        param_grid["opt_params"] = [{}]

    keys = list(param_grid.keys())
    raw_combinations = list(product(*(param_grid[key] for key in keys)))
    combinations = []
    for combo in raw_combinations:
        combo_dict = dict(zip(keys, combo))
        combo_dict = __filter_optimizer_params(combo_dict)
        combinations.append(combo_dict)

    # Remove duplicate combinations
    unique_combinations = []
    for combo in combinations:
        if combo not in unique_combinations:
            unique_combinations.append(combo)
    return unique_combinations


def __filter_optimizer_combinations(
    combinations: List[Dict[str, Any]], optimizers: List[Any]
    ) -> List[Dict[str, Any]]:
    """
    Filter out parameter combinations with empty optimizer parameters if non-empty parameters
    exist for the same optimizer.

    Parameters
    ----------
    combinations : list of dict
        List of parameter combinations.
    optimizers : list
        List of optimizer identifiers from the parameter grid.

    Returns
    -------
    list of dict
        Filtered list of parameter combinations.
    """
    # Track for each optimizer if a combination with non-empty parameters exists
    optimizer_has_params = {opt: False for opt in optimizers}
    for combo in combinations:
        optimizer = combo.get("optimizer", None)
        if optimizer is not None and combo.get("opt_params", {}) != {}:
            optimizer_has_params[optimizer] = True

    filtered_combos = []
    for combo in combinations:
        optimizer = combo.get("optimizer", None)
        # If there are non-empty parameters for this optimizer, skip empty ones
        if optimizer is not None and combo.get("opt_params", {}) == {} and optimizer_has_params[optimizer]:
            continue
        filtered_combos.append(combo)
    return filtered_combos


def accuracy_scoring(model: Model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """
    Default scoring function to calculate accuracy.

    Parameters
    ----------
    model : Model
        Trained model used for predictions.
    X_val : np.ndarray
        Validation feature data.
    y_val : np.ndarray
        Validation labels (assumed to be one-hot encoded).

    Returns
    -------
    float
        Accuracy score computed as the mean of correct predictions.
    """
    predictions = model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_val, axis=1)
    return np.mean(predicted_classes == true_classes)


def grid_search_cv(
    model_class: Type[Model],
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cv: int = 3,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    scoring: str = "accuracy",
    verbose: int = 1
    ) -> Dict[str, Any]:
    """
    Perform grid search cross-validation to tune hyperparameters and select the best model configuration.

    Parameters
    ----------
    model_class : Model
        The model class to be trained.
    param_grid : dict
        Dictionary containing hyperparameters to search over. Each key should map to a list of possible values.
    X : np.ndarray
        Feature dataset.
    y : np.ndarray
        Target labels.
    validation_data : tuple, optional
        Tuple (X_val, y_val) for final model validation; by default None.
    cv : int, optional
        Number of cross-validation folds; by default 3.
    shuffle : bool, optional
        Whether to shuffle the dataset before splitting; by default True.
    random_state : int, optional
        Seed for random number generators; by default None.
    scoring : callable, optional
        Scoring function to evaluate model performance; by default uses `default_scoring`.
    verbose : int, optional
        Verbosity level for printing progress; by default 1.

    Returns
    -------
    dict
        Dictionary containing:
            - best_score: Best cross-validation score.
            - best_params: Best hyperparameter configuration.
            - best_model: Model trained on the full dataset with best parameters.
            - best_loss_history: Training loss history for the best model.
            - results: List of tuples (params, mean_score, std_score) for each configuration.
    """
    
    # CV check to bound the splits between 2 and the number of samples.
    if cv < 2:
        raise ValueError(f"The parameter 'cv' cannot be less than 2.")
    
    if cv > X.shape[0]:
        raise ValueError(f"The parameter 'cv' cannot be more than the number of samples.")
    
    # Create a random generator with a fixed seed if random_state is set
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    n_samples = X.shape[0]

    # Shuffle the dataset if required
    if shuffle:
        perm = np.random.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    # Create folds for cross-validation
    folds = __create_folds(n_samples, cv)

    # Use the default scoring function if none is provided
    scoring_name = scoring
    
    if scoring_name not in SCORING_FUNCTIONS:
        raise ValueError(f"'{scoring_name}' is not a valid score function.")
    
    scoring = SCORING_FUNCTIONS[scoring_name]["fn"]

    # Generate all parameter combinations from the grid
    combinations = __generate_param_combinations(param_grid)

    # Filter out optimizer combinations if needed
    optimizers = param_grid.get("optimizer", [])
    if optimizers:
        combinations = __filter_optimizer_combinations(combinations, optimizers)

    if verbose:
        print("Parameter combinations to evaluate:")
        for combo in combinations:
            print(combo)
        print(f"Total combinations: {len(combinations)}\n")

    best_score = SCORING_FUNCTIONS[scoring_name]["best_init"]
    best_params = None
    best_model = None
    results = []

    config_counter = 1
    # Evaluate each parameter combination using cross-validation
    for params in combinations:
        if verbose:
            print(f"Training configuration [{config_counter}/{len(combinations)}]: {params}")

        fold_scores = []

        # Iterate over each fold
        for fold_index, (start_idx, end_idx) in enumerate(folds):
            if verbose:
                print(f"\nFold {fold_index + 1}/{len(folds)}\n")

            # Split into training and validation sets for the current fold
            X_val_fold = X[start_idx:end_idx]
            y_val_fold = y[start_idx:end_idx]
            X_train_fold = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
            y_train_fold = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)

            # Initialize model using only valid parameters for the constructor
            model_init_params = __extract_model_init_params(model_class, params)
            model = model_class(**model_init_params)

            # Add layers if provided
            if "layers_config" in params:
                for layer in copy.deepcopy(params["layers_config"]):
                    model.add(layer)

            # Get training hyperparameters
            epochs = params.get("epochs", 25)
            batch_size = params.get("batch_size", 32)
            patience = params.get("patience", epochs)
            shuffle_local = params.get("shuffle", True)

            # Train the model on the current training fold
            model.fit(
                X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                shuffle=shuffle_local,
                patience=patience,
                verbose=verbose,
                random_state=random_state
            )

            # Evaluate the model on the validation fold
            score = scoring(model, X_val_fold, y_val_fold)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results.append((params, mean_score, std_score))

        if verbose:
            print(f"Configuration: {params}")
            print(f"Mean score ({scoring_name}): {mean_score:.4f}, Std: {std_score:.4f}\n")

        # Update best parameters if the current configuration is better
        if SCORING_FUNCTIONS[scoring_name]["compare"](mean_score, best_score):
            best_score = mean_score
            best_params = params

        config_counter += 1

    if best_params is None:
        raise ValueError("No valid hyperparameter combination found.")

    # Train the final model on the full dataset with the best hyperparameters
    final_model_params = __extract_model_init_params(model_class, best_params)
    best_model = model_class(**final_model_params)

    if "layers_config" in best_params:
        for layer in copy.deepcopy(best_params["layers_config"]):
            best_model.add(layer)

    epochs = best_params.get("epochs", 25)
    batch_size = best_params.get("batch_size", 32)
    patience = best_params.get("patience", epochs)
    shuffle_local = best_params.get("shuffle", True)

    if verbose:
        print("Training final model on full dataset with best parameters...\n")

    best_loss_history = best_model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        shuffle=shuffle_local,
        patience=patience,
        verbose=verbose,
        random_state=random_state
    )

    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_model": best_model,
        "best_loss_history": best_loss_history,
        "results": results
    }



# TODO: Make adjustments in relation to grid search fixes.

def random_search_cv(model_class, param_grid, X, y, n_iter, cv=3, shuffle=True, random_state=None, scoring=None, verbose=1):
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    n_samples = X.shape[0]
    if shuffle:
        perm = np.random.permutation(n_samples)
        X = X[perm]
        y = y[perm]
    
    fold_size = math.ceil(n_samples / cv)
    folds = []
    start = 0
    for _ in range(cv):
        end = min(start + fold_size, n_samples)
        folds.append((start, end))
        start = end

    def default_scoring(model, X_val, y_val):
        y_pred = model.predict(X_val)
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_val, axis=1)
        return np.mean(pred_classes == true_classes)
    
    if scoring is None:
        scoring = default_scoring

    best_score = -float("inf")
    best_params = None
    best_model = None
    best_loss_history = None
    results = []

    param_keys = list(param_grid.keys())
    
    for i in range(n_iter):
        combo_dict = {k: random.choice(param_grid[k]) for k in param_keys}
        if verbose:
            print(f"\nIteration {i+1}/{n_iter}")
            print("Testing configuration:")
            print(combo_dict)

        fold_scores = []
        for fold_i, (start_idx, end_idx) in enumerate(folds):
            if verbose:
                print(f"\n  Fold {fold_i+1}/{len(folds)}")
            X_val_fold = X[start_idx:end_idx]
            y_val_fold = y[start_idx:end_idx]
            X_train_fold = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
            y_train_fold = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)

            model_init_params = {
                "loss_fn": combo_dict["loss_fn"],
                "optimizer": combo_dict["optimizer"],
                "opt_params": combo_dict["opt_params"]
            }
            model = model_class(**model_init_params)
            for layer in combo_dict["layers_config"]:
                model.add(layer)

            epochs = combo_dict.get("epochs", 25)
            batch_size = combo_dict.get("batch_size", 32)
            patience = combo_dict.get("patience", epochs)
            shuffle_local = combo_dict.get("shuffle", True)

            model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size,
                      validation_data=None, shuffle=shuffle_local, patience=patience, verbose=0)
            score = scoring(model, X_val_fold, y_val_fold)
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        results.append((combo_dict, mean_score, std_score))
        if verbose:
            print(f"Iteration {i+1} Score: {mean_score:.3f} ± {std_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = combo_dict
            final_model = model_class(loss_fn=combo_dict["loss_fn"],
                                      optimizer=combo_dict["optimizer"],
                                      opt_params=combo_dict["opt_params"])
            for layer in combo_dict["layers_config"]:
                final_model.add(layer)
            best_loss_history = final_model.fit(X, y, epochs=epochs, batch_size=batch_size,
                                                validation_data=None, shuffle=shuffle_local, patience=patience, verbose=0)
            best_model = final_model

    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_model": best_model,
        "best_loss_history": best_loss_history,
        "results": results
    }
