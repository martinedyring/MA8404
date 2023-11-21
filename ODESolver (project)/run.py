# Import useful packages and pre-defined helper functions
import os
import torch
import torch.nn as nn

import utilities as utilities
import params as params
import networks as networks

# Model Activation Functions
activation_functions = [
    nn.ReLU,
    nn.Sigmoid,
    nn.SiLU,
    nn.Tanh,
    nn.ELU,
    nn.Hardsigmoid,
    nn.GELU,
]

# Model Optimizer
model_optimizer = torch.optim.SGD

# Model Loss Function
model_lossfn = nn.BCELoss

file_location = os.path.dirname(__file__)
out_folder = os.path.join(file_location, "figures")
os.makedirs(out_folder, exist_ok=True)

# Set parameters
for dataset_type, dataset_kwargs in params.dataset_kwargs_dict.items():
    out_folder_data = os.path.join(out_folder, dataset_type)
    os.makedirs(out_folder_data, exist_ok=True)

    # Load dataset (X, y)
    X, y = dataset_kwargs["get_data_function"](**dataset_kwargs)

    # Load to torch tensor
    # Train test split
    (
        train_dataloader,
        test_dataloader,
        train_data,
        test_data,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = utilities.get_dataloader_from_numpy_dataset(
        X, y, color_label_dict=dataset_kwargs["color_label_dict"], fig_show=False
    )

    # Load to torch tensor
    # Train test split
    (
        train_dataloader,
        test_dataloader,
        train_data,
        test_data,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = utilities.get_dataloader_from_numpy_dataset(
        X,
        y,
        color_label_dict=params.color_label_dict,
        fig_save=True,
        fig_fname=os.path.join(out_folder_data, f"{dataset_type}_dataset.png"),
    )

    # Get dim of data
    input_dim = X.shape[-1]
    if len(y.shape) == 1:
        output_dim = 1
    else:
        output_dim = y.shape[-1]

    for ode_varaint, ode_variant_kwargs in params.ode_kwargs_dict.items():
        for (
            solver_method,
            solver_method_kwargs,
        ) in params.solver_method_kwargs_dict.items():
            # Training
            setup_kwargs = {
                "ModelODE": networks.NeuralODE,
                "model_lossfn": model_lossfn,
                "model_optimizer": model_optimizer,
                "n_epochs": params.n_epochs,
            }

            data_kwargs = {
                "train_dataloader": train_dataloader,
                "test_dataloader": test_dataloader,
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }

            plotting_kwargs = {
                # "fig_show": True,
                "fig_save": True,
                "fig_fname": os.path.join(
                    out_folder_data, f"{ode_varaint}_{solver_method}_{dataset_type}.png"
                ),
            }
            model_object_dict = utilities.run_model(
                activation_functions=activation_functions,
                **ode_variant_kwargs,
                **solver_method_kwargs,
                **setup_kwargs,
                **data_kwargs,
                **dataset_kwargs,
                **plotting_kwargs,
            )

            # Plot transformation of X points
            for activation_function in activation_functions:
                act_name = activation_function.__name__
                plot_model = model_object_dict[act_name]

                out_folder_data_act = os.path.join(out_folder_data, act_name)
                os.makedirs(out_folder_data_act, exist_ok=True)

                # 3D plot for transformation with time stepping
                utilities.plot_transformation_3d(
                    x_transformed_reduced=plot_model.x_transformed_reduced,
                    color_transformed_reduced=plot_model.color_transformed_reduced,
                    static=False,
                    interactive=True,
                    fig_save=True,
                    fig_fname=os.path.join(
                        out_folder_data_act,
                        f"{ode_varaint}_{solver_method}_{dataset_type}_{act_name}_3d.png",
                    ),
                )

                # 2D plot for transformation with time stepping
                utilities.plot_transformation_2d(
                    x_transformed_reduced=plot_model.x_transformed_reduced,
                    color_transformed_reduced=plot_model.color_transformed_reduced,
                    show_decision_boundary=True,
                    model=plot_model,
                    # FIXME: more columns for multiple time steps
                    # num_col=3, 6,
                    num_col=3 if ode_variant_kwargs["num_hidden_layers"] < 7 else 7,
                    fig_save=True,
                    fig_fname=os.path.join(
                        out_folder_data_act,
                        f"{ode_varaint}_{solver_method}_{dataset_type}_{act_name}_2d.png",
                    ),
                )

                # Loss, accuracy and confusion matrix
                utilities.plot_evaluation_score(
                    plot_model.train_loss_per_epoch,
                    plot_model.test_loss_per_epoch,
                    plot_model.train_acc_per_epoch,
                    plot_model.test_acc_per_epoch,
                    plot_model.confusion_matrix,
                    fig_save=True,
                    fig_fname=os.path.join(
                        out_folder_data_act,
                        f"{ode_varaint}_{solver_method}_{dataset_type}_{act_name}_metrix.png",
                    ),
                )
