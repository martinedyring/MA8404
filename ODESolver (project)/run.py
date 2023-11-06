# Import useful packages and pre-defined helper functions
import torch
import torch.nn as nn
import os

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
for dataset_type in params.dataset_types:
    out_folder_data = os.path.join(out_folder, dataset_type)
    os.makedirs(out_folder_data, exist_ok=True)

    # Load and prepare dataset
    # Load dataset (X, y)
    if dataset_type == "spiral":
        # Sprial data
        X, y = utilities.generate_spiral_data(params.n_points, noise=0.8, degree=540)

    elif dataset_type == "circle":
        # Circle data
        X, y = utilities.generate_circle_data(params.n_points, noise=0.1, factor=0.8)

    else:
        raise ValueError(
            f"Dataset type {dataset_type} is not defined. Please try another type!"
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

    for models_kwargs_dict in [params.shallow_kwargs_dict, params.deep_kwargs_dict]:
        for name, model_kwargs in models_kwargs_dict.items():
            # Training
            setup_kwargs = {
                "ModelODE": networks.NeuralODE,
                "model_lossfn": model_lossfn,
                "model_optimizer": model_optimizer,
                "input_dim": input_dim,
                "output_dim": output_dim,
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
                    out_folder_data,
                    f"{model_kwargs['name']}_{dataset_type}_all.png",
                ),
            }
            model_object_dict = utilities.run_model(
                activation_functions=activation_functions,
                **model_kwargs,
                **setup_kwargs,
                **data_kwargs,
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
                        f"{model_kwargs['name']}_{dataset_type}_{act_name}_3d.png",
                    ),
                )

                # 2D plot for transformation with time stepping
                utilities.plot_transformation_2d(
                    x_transformed_reduced=plot_model.x_transformed_reduced,
                    color_transformed_reduced=plot_model.color_transformed_reduced,
                    show_decision_boundary=True,
                    model=plot_model,
                    fig_save=True,
                    fig_fname=os.path.join(
                        out_folder_data_act,
                        f"{model_kwargs['name']}_{dataset_type}_{act_name}_2d.png",
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
                        f"{model_kwargs['name']}_{dataset_type}_{act_name}_metrix.png",
                    ),
                )
