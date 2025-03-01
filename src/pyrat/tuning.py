import inspect
import numpy as np
import math
import random
import copy
from itertools import product

from . import optimizers as rat_opt

def __filter_opt_params(combo_dict):
    if "optimizer" in combo_dict and "opt_params" in combo_dict:
        selected_opt = rat_opt.OPTIMIZERS.get(combo_dict["optimizer"])
        if selected_opt is not None:
            opt_class, _ = selected_opt
            valid_opt_params = [k for k in inspect.signature(opt_class.__init__).parameters if k != "self"]
            combo_dict["opt_params"] = {k: v for k, v in combo_dict["opt_params"].items() if k in valid_opt_params}
    return combo_dict

def __get_model_init_params(model_class, combo_dict):
    constructor_params = inspect.signature(model_class.__init__).parameters
    model_init_params = {}
    for key in constructor_params:
        if key == "self":
            continue
        if key in combo_dict:
            model_init_params[key] = combo_dict[key]
    return model_init_params

# TODO: Add support to other scoring functions. General refactoring.

def grid_search_cv(model_class, param_grid, X, y, validation_data=None, cv=3, shuffle=True, random_state=None, scoring=None, verbose=1):
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

    param_keys = list(param_grid.keys())
    raw_combos = list(product(*[param_grid[k] for k in param_keys]))

    # TODO: If there is a combo for an optimizer with parameters, it overrides the one without parameters.

    filtered_combos = []
    for combo_tuple in raw_combos:
        combo_dict = dict(zip(param_keys, combo_tuple))
        filtered_dict = __filter_opt_params(combo_dict)
        filtered_combos.append(filtered_dict)

    unique_combos = []
    for c in filtered_combos:
        if c not in unique_combos:
            unique_combos.append(c)

    combos = unique_combos

    best_score = -float("inf")
    best_params = None
    best_model = None
    results = []

    config_counter = 1

    for combo_dict in combos:
        if verbose:
            print(f"\nTraining configuration [{config_counter}/{len(combos)}]:\n{combo_dict}")

        fold_scores = []

        for fold_i, (start_idx, end_idx) in enumerate(folds):
            if verbose:
                print(f"\nFold {fold_i+1}/{len(folds)}")

            X_val_fold = X[start_idx:end_idx]
            y_val_fold = y[start_idx:end_idx]
            X_train_fold = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
            y_train_fold = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)

            model_init_params = __get_model_init_params(model_class, combo_dict)
            model = model_class(**model_init_params)

            if "layers_config" in combo_dict:
                for layer in copy.deepcopy(combo_dict["layers_config"]):
                    model.add(layer)

            epochs = combo_dict.get("epochs", 25)
            batch_size = combo_dict.get("batch_size", 32)
            patience = combo_dict.get("patience", epochs)
            shuffle_local = combo_dict.get("shuffle", True)

            model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, validation_data=(X_val_fold, y_val_fold), shuffle=shuffle_local, patience=patience, verbose=verbose)

            score = scoring(model, X_val_fold, y_val_fold)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        results.append((combo_dict, mean_score, std_score))
        if verbose:
            print(f"\nConfiguration: {combo_dict}\nMean score: {mean_score:.4f}, Std: {std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = combo_dict
            
        config_counter += 1

    if best_params is None:
        raise ValueError("No valid combination found (best_params is None).")

    final_model_init_params = __get_model_init_params(model_class, best_params)
    best_model = model_class(**final_model_init_params)

    if "layers_config" in best_params:
        for layer in copy.deepcopy(best_params["layers_config"]):
            best_model.add(layer)

    epochs = best_params.get("epochs", 25)
    batch_size = best_params.get("batch_size", 32)
    patience = best_params.get("patience", epochs)
    shuffle_local = best_params.get("shuffle", True)

    if verbose:
        print("\nTraining final model on full dataset with best parameters...\n")

    final_loss_history = best_model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, shuffle=shuffle_local, patience=patience, verbose=verbose)

    best_loss_history = final_loss_history

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
