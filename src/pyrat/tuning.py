import inspect
import numpy as np
import math
import random
import copy
import time
from itertools import product
from joblib import Parallel, delayed
from typing import Any, Dict, List, Optional, Tuple, Type

from .models import Model
from .optimizers import OPTIMIZERS
from .functions import SCORING_FUNCTIONS


def __extract_model_init_params(model_class: Type[Model], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts valid constructor parameters for the given model class from a parameter dictionary.

    Parameters
    ----------
    model_class : Type[Model]
        The model class to be instantiated.
    params : Dict[str, Any]
        The dictionary containing potential parameters.

    Returns
    -------
    Dict[str, Any]
        A filtered dictionary containing only the parameters that match the model class constructor signature.
    """
    constructor_params = inspect.signature(model_class.__init__).parameters
    valid = {}
    for k in constructor_params:
        if k != "self" and k in params:
            valid[k] = params[k]
    return valid


def __build_optimizer(params: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Builds the optimizer based on the 'optimizer' key in params and extracts its valid parameters.

    Parameters
    ----------
    params : Dict[str, Any]
        A dictionary of parameters that may contain an 'optimizer' key and optimizer hyperparameters.

    Returns
    -------
    Tuple[Optional[str], Dict[str, Any]]
        A tuple (optimizer_name, optimizer_params). If no valid optimizer is found, returns (None, {}).
    """
    name = params.get("optimizer", None)
    if name not in OPTIMIZERS:
        return None, {}
    cls = OPTIMIZERS[name]
    valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    extracted = {}
    # Extract only the hyperparameters relevant to the optimizer's constructor
    for k in list(params.keys()):
        if k in valid_keys:
            extracted[k] = params.pop(k)
    return name, extracted


def __create_folds(n_samples: int, cv: int) -> List[Tuple[int, int]]:
    """
    Splits dataset indices into cv folds.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    cv : int
        Number of folds.

    Returns
    -------
    List[Tuple[int, int]]
        List of (start_index, end_index) for each fold.
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
    Generates all parameter combinations from the given param_grid via Cartesian product.

    Parameters
    ----------
    param_grid : Dict[str, List[Any]]
        A dictionary where keys are parameter names, and values are lists of possible parameter values.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries, each representing one unique combination of parameters.
    """
    keys = list(param_grid.keys())
    all_products = product(*(param_grid[k] for k in keys))
    combos = []
    for combo in all_products:
        d = dict(zip(keys, combo))
        combos.append(d)
    unique = []
    for combo in combos:
        if combo not in unique:
            unique.append(combo)
    return unique


def grid_search_cv(
    model_class: Type[Model],
    param_grid: Dict[str, List[Any]],
    X: np.ndarray,
    y: np.ndarray,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cv: int = 3,
    random_state: Optional[int] = None,
    scoring: str = "accuracy",
    verbose: int = 1,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Performs a grid search cross-validation over a parameter grid.

    Parameters
    ----------
    model_class : Type[Model]
        The model class to be instantiated and trained.
    param_grid : Dict[str, List[Any]]
        Dictionary with parameters as keys and lists of parameter values as values.
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    validation_data : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Validation set as (X_val, y_val), by default None.
    cv : int, optional
        Number of folds for cross-validation, by default 3.
    random_state : Optional[int], optional
        Random seed, by default None.
    scoring : str, optional
        Name of the scoring function (key in SCORING_FUNCTIONS), by default "accuracy".
    verbose : int, optional
        Verbosity level (0 = no prints, >0 = prints), by default 1.
    n_jobs : int, optional
        Number of parallel processes, by default 1.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "best_score": the best score found
        - "best_params": the parameters that yielded the best score
        - "best_model": the model retrained on the entire dataset
        - "best_loss_history": training loss history of the final model
        - "best_val_loss_history": validation loss history of the final model
        - "best_accuracy_history": accuracy history of the final model (if any)
        - "best_val_accuracy_history": validation accuracy history of the final model (if any)
        - "results": a list of training results for each combination
        - "search_time": the total search time in seconds
    """
    if cv < 2:
        raise ValueError("cv must be >= 2.")
    if cv > X.shape[0]:
        raise ValueError("cv cannot exceed the number of samples.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    start_time_total = time.time()
    folds = __create_folds(X.shape[0], cv)

    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"{scoring} is not valid.")
    scoring_fn = SCORING_FUNCTIONS[scoring]["fn"]
    best_val = SCORING_FUNCTIONS[scoring]["best_init"]

    combos = __generate_param_combinations(param_grid)

    if verbose and n_jobs == 1:
        print("Parameter combinations to evaluate:")
        for combo in combos:
            print(combo)
    if verbose:
        print(f"Total combinations: {len(combos)}\n")

    best_score = best_val
    best_params = None
    best_model = None
    results = []

    def _train_and_eval(params: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float, float]:
        """
        Trains and evaluates one configuration across all folds.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters for the current configuration.

        Returns
        -------
        Tuple[Dict[str, Any], float, float, float]
            (params, mean_score, std_score, elapsed_time)
        """
        if verbose > 0:
            print(f"Starting training for configuration: {params}")

        t0 = time.time()
        fold_scores = []
        local_verbose = 0 if (n_jobs > 1) else verbose

        pcopy = dict(params)
        opt_name, opt_conf = __build_optimizer(pcopy)
        m_init = __extract_model_init_params(model_class, pcopy)
        m = model_class(**m_init)

        if opt_name is not None:
            m.optimizer = OPTIMIZERS[opt_name](**opt_conf)

        if "layers_config" in pcopy:
            for layer in copy.deepcopy(pcopy["layers_config"]):
                m.add(layer)

        epochs = pcopy.get("epochs", 25)
        batch_size = pcopy.get("batch_size", 32)
        patience = pcopy.get("patience", epochs)
        shuffle_local = pcopy.get("shuffle", True)

        for fold_index, (start_idx, end_idx) in enumerate(folds):
            if verbose > 0 and n_jobs == 1:
                print(f"Training configuration fold {fold_index + 1}/{cv} ...")

            X_val_fold = X[start_idx:end_idx]
            y_val_fold = y[start_idx:end_idx]
            X_train_fold = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
            y_train_fold = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)

            m2 = copy.deepcopy(m)
            m2.fit(
                X_train_fold,
                y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                shuffle=shuffle_local,
                patience=patience,
                verbose=local_verbose,
                random_state=random_state
            )
            fold_score = scoring_fn(m2, X_val_fold, y_val_fold)
            fold_scores.append(fold_score)

        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        elapsed = time.time() - t0
        return params, mean_s, std_s, elapsed


    if n_jobs == 1:
        print()
        for i, combo in enumerate(combos, 1):
            print(f"Training configuration [{i}/{len(combos)}] ...")
            c2, ms, ss, tt = _train_and_eval(combo)
            print(f"Training finished in {tt:.2f} seconds.")
            results.append({"params": c2, "mean_score": ms, "std_score": ss, "train_time": tt})

            if verbose:
                print(f"Mean score={ms:.4f}, std={ss:.4f}\n")

            if SCORING_FUNCTIONS[scoring]["compare"](ms, best_score):
                best_score = ms
                best_params = c2
    else:
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(_train_and_eval)(combo) for combo in combos
        )
        print()
        for i, (pp, ms, ss, tt) in enumerate(parallel_results, 1):
            print(f"[{i}/{len(combos)}] Training finished in {tt:.2f} seconds.")
            results.append({"params": pp, "mean_score": ms, "std_score": ss, "train_time": tt})
            if verbose:
                print(f"Mean score={ms:.4f}, std={ss:.4f}\n")
            if SCORING_FUNCTIONS[scoring]["compare"](ms, best_score):
                best_score = ms
                best_params = pp

    if best_params is None:
        raise ValueError("No valid combination found.")

    final_copy = dict(best_params)
    opt_name, opt_conf = __build_optimizer(final_copy)
    m_init = __extract_model_init_params(model_class, final_copy)
    best_model = model_class(**m_init)
    if opt_name is not None:
        best_model.optimizer = OPTIMIZERS[opt_name](**opt_conf)
    if "layers_config" in final_copy:
        for layer in copy.deepcopy(final_copy["layers_config"]):
            best_model.add(layer)

    epochs = final_copy.get("epochs", 25)
    batch_size = final_copy.get("batch_size", 32)
    patience = final_copy.get("patience", epochs)
    shuffle_local = final_copy.get("shuffle", True)

    if verbose:
        print("Retraining final model on full dataset with best parameters...\n")

    final_info = best_model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        shuffle=shuffle_local,
        patience=patience,
        verbose=verbose,
        random_state=random_state
    )
    best_loss_history = final_info.get("loss", [])
    best_val_loss_history = final_info.get("val_loss", [])
    best_accuracy_history = final_info.get("accuracy", [])
    best_val_accuracy_history = final_info.get("val_accuracy", [])

    total_time = time.time() - start_time_total
    if verbose:
        print(f"\nTotal grid search execution time: {total_time:.2f} second(s).\n")

    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_model": best_model,
        "best_loss_history": best_loss_history,
        "best_val_loss_history": best_val_loss_history,
        "best_accuracy_history": best_accuracy_history,
        "best_val_accuracy_history": best_val_accuracy_history,
        "results": results,
        "search_time": total_time
    }


def random_search_cv(
    model_class: Type[Model],
    param_grid: Dict[str, List[Any]],
    n_iter: int,
    X: np.ndarray,
    y: np.ndarray,
    validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cv: int = 3,
    random_state: Optional[int] = None,
    scoring: str = "accuracy",
    verbose: int = 1,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Performs a random search cross-validation by sampling configurations from a parameter grid.

    Parameters
    ----------
    model_class : Type[Model]
        The model class to be instantiated and trained.
    param_grid : Dict[str, List[Any]]
        Dictionary of parameters as keys and lists of possible values as values.
    n_iter : int
        Number of random configurations to sample and evaluate.
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    validation_data : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Validation set as (X_val, y_val), by default None.
    cv : int, optional
        Number of folds for cross-validation, by default 3.
    random_state : Optional[int], optional
        Random seed, by default None.
    scoring : str, optional
        Name of the scoring function, by default "accuracy".
    verbose : int, optional
        Verbosity level (0 = no prints, >0 = prints), by default 1.
    n_jobs : int, optional
        Number of parallel processes, by default 1.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "best_score": the best score found
        - "best_params": the parameters that yielded the best score
        - "best_model": the model retrained on the entire dataset
        - "best_loss_history": training loss history of the final model
        - "best_val_loss_history": validation loss history of the final model
        - "best_accuracy_history": accuracy history of the final model (if any)
        - "best_val_accuracy_history": validation accuracy history of the final model (if any)
        - "results": a list of training results for each combination
        - "search_time": the total search time in seconds
    """
    if cv < 2:
        raise ValueError("cv must be >= 2.")
    if cv > X.shape[0]:
        raise ValueError("cv cannot exceed the number of samples.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    if verbose:
        print(f"Total random configurations: {n_iter}\n")

    start_time_total = time.time()
    folds = __create_folds(X.shape[0], cv)
    if scoring not in SCORING_FUNCTIONS:
        raise ValueError(f"{scoring} is not valid.")
    scoring_fn = SCORING_FUNCTIONS[scoring]["fn"]
    best_val = SCORING_FUNCTIONS[scoring]["best_init"]
    best_score = best_val
    best_params = None
    results = []

    def sample_params() -> Dict[str, Any]:
        """
        Randomly picks a single parameter value for each key in param_grid.

        Returns
        -------
        Dict[str, Any]
            One dictionary representing a randomly sampled configuration.
        """
        p = {}
        for k, v in param_grid.items():
            p[k] = random.choice(v)
        return p

    def _train_and_eval(params: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float, float]:
        """
        Trains and evaluates one configuration across all folds.

        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameters for the current configuration.

        Returns
        -------
        Tuple[Dict[str, Any], float, float, float]
            (params, mean_score, std_score, elapsed_time)
        """
        if verbose > 0:
            print(f"Starting training for configuration: {params}")

        t0 = time.time()
        fold_scores = []
        local_verbose = 0 if (n_jobs > 1) else verbose

        pcopy = dict(params)
        opt_name, opt_conf = __build_optimizer(pcopy)
        m_init = __extract_model_init_params(model_class, pcopy)
        m = model_class(**m_init)

        if opt_name is not None:
            m.optimizer = OPTIMIZERS[opt_name](**opt_conf)

        if "layers_config" in pcopy:
            for layer in copy.deepcopy(pcopy["layers_config"]):
                m.add(layer)

        epochs = pcopy.get("epochs", 25)
        batch_size = pcopy.get("batch_size", 32)
        patience = pcopy.get("patience", epochs)
        shuffle_local = pcopy.get("shuffle", True)

        for fold_index, (start_idx, end_idx) in enumerate(folds):
            if verbose > 0 and n_jobs == 1:
                print(f"Training configuration fold {fold_index + 1}/{cv} ...")

            X_val_fold = X[start_idx:end_idx]
            y_val_fold = y[start_idx:end_idx]
            X_train_fold = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
            y_train_fold = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)

            m2 = copy.deepcopy(m)
            m2.fit(
                X_train_fold,
                y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                shuffle=shuffle_local,
                patience=patience,
                verbose=local_verbose,
                random_state=random_state
            )
            fold_score = scoring_fn(m2, X_val_fold, y_val_fold)
            fold_scores.append(fold_score)

        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        elapsed = time.time() - t0
        return params, mean_s, std_s, elapsed


    if n_jobs == 1:
        for i in range(1, n_iter + 1):
            combo = sample_params()
            print(f"Training configuration [{i}/{n_iter}] ...")
            pp, ms, ss, tt = _train_and_eval(combo)
            print(f"[{i}/{n_iter}] Training finished in {tt:.2f} seconds.")
            results.append({"params": pp, "mean_score": ms, "std_score": ss, "train_time": tt})
            if verbose:
                print(f"Mean score={ms:.4f}, std={ss:.4f}\n")

            if SCORING_FUNCTIONS[scoring]["compare"](ms, best_score):
                best_score = ms
                best_params = pp
    else:
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(_train_and_eval)(sample_params()) for _ in range(n_iter)
        )
        print()
        for i, (pp, ms, ss, tt) in enumerate(parallel_results, 1):
            print(f"[{i}/{n_iter}] Training finished in {tt:.2f} seconds.")
            results.append({"params": pp, "mean_score": ms, "std_score": ss, "train_time": tt})
            if verbose:
                print(f"Mean score={ms:.4f}, std={ss:.4f}\n")

            if SCORING_FUNCTIONS[scoring]["compare"](ms, best_score):
                best_score = ms
                best_params = pp

    if best_params is None:
        raise ValueError("No valid configuration found.")

    final_copy = dict(best_params)
    opt_name, opt_conf = __build_optimizer(final_copy)
    m_init = __extract_model_init_params(model_class, final_copy)
    best_model = model_class(**m_init)
    if opt_name is not None:
        best_model.optimizer = OPTIMIZERS[opt_name](**opt_conf)
    if "layers_config" in final_copy:
        for layer in copy.deepcopy(final_copy["layers_config"]):
            best_model.add(layer)

    epochs = final_copy.get("epochs", 25)
    batch_size = final_copy.get("batch_size", 32)
    patience = final_copy.get("patience", epochs)
    shuffle_local = final_copy.get("shuffle", True)

    if verbose:
        print("Retraining final model on full dataset with best parameters...\n")

    final_info = best_model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        shuffle=shuffle_local,
        patience=patience,
        verbose=verbose,
        random_state=random_state
    )
    best_loss_history = final_info.get("loss", [])
    best_val_loss_history = final_info.get("val_loss", [])
    best_accuracy_history = final_info.get("accuracy", [])
    best_val_accuracy_history = final_info.get("val_accuracy", [])
    
    total_time = time.time() - start_time_total
    if verbose:
        print(f"\nTotal random search execution time: {total_time:.2f} second(s).\n")

    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_model": best_model,
        "best_loss_history": best_loss_history,
        "best_val_loss_history": best_val_loss_history,
        "best_accuracy_history": best_accuracy_history,
        "best_val_accuracy_history": best_val_accuracy_history,
        "results": results,
        "search_time": total_time
    }
