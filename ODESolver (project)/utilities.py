# Helper function to neural_ode

import torch
import numpy as np
import sklearn
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.classification import BinaryAccuracy
from tqdm.auto import tqdm
from sklearn.metrics import auc
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import seaborn as sns
from plotly.subplots import make_subplots

# global variables
import params as params


def train_model(
    model,
    data_loader,
    loss_fn,
    optimizer,
    verbose=False,
):
    """
    Training loop for 1 epoch

    Args:
        model (_type_): _description_
        data_loader (_type_): _description_
        loss_fn (_type_): _description_
        optimizer (_type_): _description_
    """
    train_loss = 0.0  # keep track of total loss
    train_acc = 0.0  # keep track of the total accuracy

    # print(len(data_loader))
    for batch, (X_data, y_data) in enumerate(data_loader):
        model.train()  # Set to training mode
        # Forward pass - make a prediction
        # print("before fp")
        y_prob, x_transformed = model(X_data)  # (model outputs probabilities)
        # print("after fp")
        y_pred = torch.round(y_prob)  # (probabilities -> binary prediction)
        # note: perform sigmoid on the "logits" dimension, not "batch" dimension
        # (in this case we have a batch size of 1, so can perform on dim=0)
        # print(f"batch: {batch}")
        # Compute the loss
        # print(y_prob)
        # print(y_data)
        loss = loss_fn(y_prob, y_data)
        train_loss += loss
        if y_pred.shape == y_data.shape:
            acc = torch.eq(y_pred, y_data).sum().item() / len(y_data) * 100
        else:
            acc = torch.eq(y_pred.argmax(dim=1), y_data).sum().item() / len(y_data) * 100
        train_acc += acc

        # Optimizer zero grad
        optimizer.zero_grad()

        # Compute gradient of the loss w.r.t. to the parameters
        loss.backward()

        # Optimizer will modify the parameters by consideting the gradient
        # Inside each parameter there is already a grad calculated,
        # this applies the optimizer algorithm to the observed parameters
        optimizer.step()

        # Print out how many samples have been seen
        if verbose:
            if batch % 400 == 0:
                print(
                    f"Looked at {batch * len(X_data)}/{len(data_loader.dataset)} samples"
                )

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    return train_loss, train_acc


def test_model(model, data_loader, loss_fn, evaluate=False):
    test_loss = 0.0  # keep track of total loss
    test_acc = 0.0  # keep track of the total accuracy
    x_transformed_array = []
    y_transformed_array = []

    model.eval()  # Set to evaluation mode

    with torch.inference_mode():  # Turn on inference context manager
        for X_test, y_test in data_loader:
            # Forward pass - Inference data
            y_prob, x_transformed = model(X_test)  # (model outputs probabilities)
            y_pred = torch.round(y_prob)  # (probabilities -> binary prediction)
            # note: perform sigmoid on the "logits" dimension, not "batch" dimension
            # (in this case we have a batch size of 1, so can perform on dim=0)

            # Compute loss and accuracy
            loss = loss_fn(y_pred, y_test)
            if y_pred.shape == y_test.shape:
                acc = torch.eq(y_pred, y_test).sum().item() / len(y_test) * 100
            else:
                acc = (
                    torch.eq(y_pred.argmax(dim=1), y_test).sum().item()
                    / len(y_test)
                    * 100
                )
            test_loss += loss
            test_acc += acc

            # Save intermidiate state
            x_transformed_array.append(x_transformed)
            y_transformed_array.append(y_test)

        # Print metrics
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        if evaluate == True:
            return (
                {
                    "model_name": model.__class__.__name__,
                    "model_loss": test_loss.item(),
                    "model_acc": test_acc,
                },
                x_transformed_array,
                y_transformed_array,
            )

        else:
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    return test_loss, test_acc


def make_predictions(model, data):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(
                sample, dim=0
            )  # Add an extra dimension and send sample to device

            # Forward pass (model outputs probability)
            pred_prob = model(sample)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob)

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


# Confusion matrix
def compute_confusion_matrix(model, dataset, data_loader):
    # Make predictions with trained model
    y_probs = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Making predictions"):
            # Do the forward pass
            y_prob, x_transformed = model(X)

            y_probs.append(y_prob)

    # Concatenate list of predictions into a tensor
    y_probs_tensor = torch.cat(y_probs)

    # Setup confusion matrix instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(dataset.classes), task="multiclass")
    confmat_tensor = confmat(preds=y_probs_tensor, target=dataset.targets)
    return confmat_tensor


# Plot decision boundaries for training and test sets
def decision_boundary_grid(model, X, y):
    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make points
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions on grid
    model.eval()
    with torch.inference_mode():
        y_grid_probs, x_transformed = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    y_grid_pred = torch.round(y_grid_probs)  # binary

    # Reshape to 2D
    y_grid_pred = y_grid_pred.reshape(xx.shape).numpy()

    return xx, yy, y_grid_pred


def get_dataloader_from_numpy_dataset(
    X, y, plot_splitting=False, color_label_dict=None, figsize=(15, 5)
):
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    print(f"Shape of synthetic dataset (X, y): {X.shape, y.shape}")

    # Train test split dataset
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42  # 20% test, 80% train
    )  # make the random split reproducible

    # Create a data loader from the dataset
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    BATCH_SIZE = 5

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,  # shuffle data every epoch
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=400,
    )

    if plot_splitting:
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        if color_label_dict is not None:
            color_dataset = np.vectorize(color_label_dict.get)(y)
            color_train = np.vectorize(color_label_dict.get)(y_train)
            color_test = np.vectorize(color_label_dict.get)(y_test)
        else:
            color_dataset = y
            color_train = y_train
            color_test = y_test

        # Plot total
        axs[0].scatter(X[:, 0], X[:, 1], c=color_dataset)
        axs[0].set_title("Total Synthetic dataset")

        # Plot train
        scatter = axs[1].scatter(
            X_train[:, 0], X_train[:, 1], c=color_train, label=y_train
        )
        legend0 = axs[1].legend(
            *scatter.legend_elements(), loc="lower left", title="Classes"
        )
        axs[1].add_artist(legend0)
        axs[1].set_title("Train dataset")

        # Plot test
        axs[2].scatter(X_test[:, 0], X_test[:, 1], c=color_test, label=y_test)
        axs[2].set_title("Test dataset")
        legend1 = axs[2].legend(
            *scatter.legend_elements(), loc="lower left", title="Classes"
        )
        axs[2].add_artist(legend1)

        plt.suptitle("Train-test split")
        plt.show()

    return (
        train_dataloader,
        test_dataloader,
        train_data,
        test_data,
        X_train,
        y_train,
        X_test,
        y_test,
    )


def generate_spiral_data(n_points, noise, degree):
    np.random.seed(42)
    n = np.sqrt(np.random.rand(n_points, 1)) * degree * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.randn(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.randn(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points), np.ones(n_points))),
    )


def generate_circle_data(n_points, noise, factor):
    # Create circles
    X, y = make_circles(
        n_samples=n_points,  # Make a large circle containing a smaller circle in 2d
        noise=noise,  # a little bit of noise to the dots
        random_state=42,
        factor=factor,
    )  # keep random state so we get the same values
    return X, y


def plot_transformation_3d(
    x_transformed_reduced, color_transformed_reduced, static=False, interactive=False
):
    if static == False and interactive == False:
        raise ValueError(
            f'At least one of the following key word argument *has* to be True: "static", "interactive"'
        )
    # Plot
    num_col = 3

    if static:
        # Initialize figure with 3D subplots (Matplotlib)
        plt_fig, axs = plt.subplots(
            1, num_col, figsize=(15, 5), subplot_kw=dict(projection="3d")
        )

    if interactive:
        # Initialize figure with 3D subplots (Plotly)
        plotly_fig = make_subplots(
            rows=1,
            cols=num_col,
            specs=[[{"type": "scene"} for _ in range(num_col)]],
            subplot_titles=tuple([f"Plot {n+1}" for n in range(num_col)]),
        )
        plotly_titels = {}

    for l in range(num_col):  # Hidden layers # Time step
        if l == num_col - 1:  # Always plot end
            l_plot = x_transformed_reduced.shape[-1] - 1
        else:
            l_plot = (
                x_transformed_reduced.shape[-1] // (num_col - 1) * l
            )  # plot every num_col step

        # fit agian for each time step
        # FIXME: pca_result = x_transformed_reduced[:, :, l_plot] hvis vi allerede har 3d
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(x_transformed_reduced[:, :, l_plot])

        if static:
            axs[l].scatter(
                xs=pca_result[:, 0],
                ys=pca_result[:, 1],
                zs=pca_result[:, 2],
                marker="o",
                c=color_transformed_reduced,
                alpha=1,
            )

            # Add subtitle
            axs[l].set_title(f"After {l_plot} time steps")

        if interactive:
            plotly_fig.add_trace(
                go.Scatter3d(
                    x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    z=pca_result[:, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=color_transformed_reduced,
                    ),
                    name=None,
                ),
                row=1,
                col=l + 1,
            )

            # Add subtitle
            plotly_titels[f"Plot {l+1}"] = f"After {l_plot} time steps"

    if static:
        plt.suptitle("Transform after each time step")
        plt.show()

    if interactive:
        plotly_fig.for_each_annotation(lambda a: a.update(text=plotly_titels[a.text]))
        plotly_fig.update_layout(title="Transformation of points")
        plotly_fig.show()


def plot_transformation_2d(
    x_transformed_reduced,
    color_transformed_reduced,
    show_decision_boundary=False,
    model=None,
):
    if show_decision_boundary:
        if model is None:
            raise ValueError(
                f"Can not plot descision boundary without model object. "
                f"Please provide model as key word arguement!"
            )
    # Use PCA to reduce dimension
    # NOTE: Dim need to be 2 in order to plot trajectory
    pca = PCA(n_components=2)
    # ex.: pca_result = pca.fit_transform(x_transformed_array[0, :, :, 0])
    # pca_result.shape : # num samples in each batch - # dimension (2)

    num_col = 3

    fig, axs = plt.subplots(1, num_col, figsize=(15, 5))

    for l in range(num_col):  # Hidden layers # Time step
        if l == num_col - 1:  # Always plot end
            l_plot = x_transformed_reduced.shape[-1] - 1
        else:
            l_plot = (
                x_transformed_reduced.shape[-1] // (num_col - 1) * l
            )  # plot every num_col step

        # fit agian for each time step
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(x_transformed_reduced[:, :, l_plot])

        # for i in range(pca_result.shape[0]):  # # num samples in each batch
        axs[l].scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            marker=".",
            c=color_transformed_reduced,
        )

        axs[l].set_title(f"After {l_plot} time steps")

        # ------------------------------- background contourf plot ----------------------------
        if show_decision_boundary:
            x_min = pca_result[:, 0].min()
            x_max = pca_result[:, 0].max()
            y_min = pca_result[:, 1].min()
            y_max = pca_result[:, 1].max()

            # create grid for prediction as coloured background
            x_grid, y_grid = np.meshgrid(
                np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000)
            )
            XGrid = torch.tensor(list(zip(x_grid.flatten(), y_grid.flatten()))).float()

            # inverse transform to match internal dimension
            XGrid_original = pca.inverse_transform(XGrid)
            XGrid_original = torch.tensor(XGrid_original).float()

            # apply linear classifier and logits to prob
            model.eval()
            # prediction = model.logits_to_prob(model.classifier(XGrid_original)).squeeze().detach().numpy()
            prediction = (
                model.logits_to_prob(model.classifier(XGrid_original))
                .squeeze()
                .detach()
                .numpy()
            )

            axs[l].contourf(
                x_grid,
                y_grid,
                prediction.reshape(1000, 1000),
                levels=50,
                cmap=plt.cm.RdYlBu_r,
                alpha=0.3,
            )

        # ------------------------------- background contourf plot ----------------------------

    plt.suptitle("Transform after each time step")
    plt.show()


def plot_evaluation_score(
    train_loss_per_epoch,
    test_loss_per_epoch,
    train_acc_per_epoch,
    test_acc_per_epoch,
    confusion_matrix,
):
    # Plot loss
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(train_loss_per_epoch, label="train")
    axs[0].plot(test_loss_per_epoch, label="validate")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss")
    axs[0].set_yscale("log")
    axs[0].legend()

    axs[1].plot(train_acc_per_epoch, label="train")
    axs[1].plot(test_acc_per_epoch, label="validate")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Training Accuracy")
    axs[1].set_yscale("log")
    axs[1].legend()

    # Plot the confusion matrix.
    axs[2] = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="g",
        xticklabels=["0", "1"],  # FIXME: change to red and blue
        yticklabels=["0", "1"],
    )
    axs[2].set_ylabel("Prediction")
    axs[2].set_xlabel("Actual")
    axs[2].set_title("Confusion matrix on test dataset")
    plt.tight_layout()
    plt.show()


def preprocess_transformed_array(x_transformed_array, y_transformed_array):
    print("Original dim (x_trans, y_trans)")
    print(np.array(x_transformed_array).shape, np.array(y_transformed_array).shape)
    print("# batches - # samples in each batch - # hidden dimension - # hidden layers\n")

    # Create reduced array
    # 1. Remove dim due to batches
    # 2. Transform form torch tensor to numpy array
    num_batches = len(x_transformed_array)
    samples_in_batch = x_transformed_array[0].shape[0]
    x_transformed_reduced = np.empty(
        [
            num_batches * samples_in_batch,
            x_transformed_array[0].shape[1],
            x_transformed_array[0].shape[2],
        ]
    )
    y_transformed_reduced = np.empty([num_batches * samples_in_batch])

    for b in range(num_batches):
        if b == num_batches - 1:
            x_transformed_reduced[samples_in_batch * b :, :, :] = x_transformed_array[b]
            y_transformed_reduced[samples_in_batch * b :] = y_transformed_array[b]
        else:
            x_transformed_reduced[
                samples_in_batch * b : samples_in_batch * (b + 1), :, :
            ] = x_transformed_array[b]
            y_transformed_reduced[
                samples_in_batch * b : samples_in_batch * (b + 1)
            ] = y_transformed_array[b]

    print(f"Reduced dim (x_trans_reduced, y_trans_reduced)")
    print(x_transformed_reduced.shape, y_transformed_reduced.shape)
    print("# samples - # hidden dimension - # hidden layers\n")

    return x_transformed_reduced, y_transformed_reduced


def run_model(activation_functions, **kwargs):
    # Unpack kwargs
    ModelODE = kwargs.get("ModelODE", None)
    input_dim = kwargs.get("input_dim", None)
    hidden_dim = kwargs.get("hidden_dim", None)
    hidden_internal_dim = kwargs.get("hidden_internal_dim", None)
    output_dim = kwargs.get("output_dim", None)
    num_hidden_layers = kwargs.get("num_hidden_layers", None)
    variant = kwargs.get("variant", None)
    method = kwargs.get("method", None)
    model_lossfn = kwargs.get("model_lossfn", None)
    model_optimizer = kwargs.get("model_optimizer", None)
    train_dataloader = kwargs.get("train_dataloader", None)
    test_dataloader = kwargs.get("test_dataloader", None)
    X_train = kwargs.get("X_train", None)
    y_train = kwargs.get("y_train", None)
    X_test = kwargs.get("X_test", None)
    y_test = kwargs.get("y_test", None)
    n_epochs = kwargs.get("n_epochs", params.n_epochs)
    color_label_dict = kwargs.get("color_label_dict", params.color_label_dict)

    # Dict to store model objects
    model_object_dict = {}

    # Initialize plot for evaluation if we are comparing different actication function
    if len(activation_functions) > 1:
        fig = plt.figure(
            constrained_layout=True, figsize=(15, 7 * len(activation_functions))
        )
        axs = fig.subfigures(len(activation_functions), 1)

    for a, activation_function in enumerate(activation_functions):
        print(f"\nActivation function {activation_function.__name__}:\n")

        # Create 1x3 subplots on each row
        if len(activation_functions) > 1:
            subaxs = axs[a].subplots(1, 3)
        else:
            fig, subaxs = plt.subplots(1, 3, figsize=(15, 5))

        # Initialize model
        torch.manual_seed(42)  # set seed for random parameters in model
        model = ModelODE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_internal_dim=hidden_internal_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,  # num time steps
            sigma=activation_function,
            variant=variant,
            method=method,
        )

        # Select optimizer and loss function
        loss_fn = model_lossfn()
        # Create an optimizer
        optimizer = model_optimizer(params=model.parameters(), lr=0.1)

        # Train and test for multiple epochs
        train_loss_per_epoch = []
        test_loss_per_epoch = []
        train_acc_per_epoch = []
        test_acc_per_epoch = []

        for epoch in range(n_epochs):
            print(f"----> EPOCH {epoch} of {n_epochs}:")
            train_loss, train_acc = train_model(
                model=model,
                data_loader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                verbose=False,
            )

            test_loss, test_acc = test_model(
                model=model, data_loader=test_dataloader, loss_fn=loss_fn
            )

            train_loss_per_epoch.append(train_loss.detach().numpy())
            train_acc_per_epoch.append(train_acc)

            test_loss_per_epoch.append(test_loss.detach().numpy())
            test_acc_per_epoch.append(test_acc)

        # Plot decision boundaries for training and test sets
        for i, (X_data, y_data, title) in enumerate(
            zip(
                [X_train.numpy(), X_test.numpy()],
                [y_train.numpy(), y_test.numpy()],
                ["Train", "Test"],
            )
        ):
            xx, yy, y_grid_pred = decision_boundary_grid(model=model, X=X_data, y=y_data)
            subaxs[i].contourf(xx, yy, y_grid_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
            subaxs[i].scatter(
                X_data[:, 0], X_data[:, 1], c=y_data, s=40, cmap=plt.cm.RdYlBu
            )
            subaxs[i].set_xlim(xx.min(), xx.max())
            subaxs[i].set_ylim(yy.min(), yy.max())
            subaxs[i].set_title(title)

        # Metrics
        probs, x_transformed = model(X_test)
        preds = torch.round(probs)
        target = y_test.type(torch.int)
        acc_metric = BinaryAccuracy()
        prc_metric = BinaryPrecisionRecallCurve()

        bprc = prc_metric(probs, target)  # Precision, recall, threshold
        subaxs[2].plot(
            bprc[0], bprc[1], label=f"AUC: {np.round(auc(bprc[1], bprc[0]), 3)}"
        )  # auc(recall, precision)
        subaxs[2].grid()
        subaxs[2].set_xlabel("Precision")
        subaxs[2].set_ylabel("Recall")
        subaxs[2].legend()
        subaxs[2].set_title("PR-curve")

        # Confusion matrix
        cm = confusion_matrix(target, preds.detach())

        try:
            axs[a].suptitle(
                f"Activation function: {activation_function.__name__}\n"
                f"w/ accuracy: {np.round(acc_metric(preds, target).item(), 3)}"
            )
        except NameError:
            plt.suptitle(
                f"Activation function: {activation_function.__name__}\n"
                f"w/ accuracy: {np.round(acc_metric(preds, target).item(), 3)}"
            )
            plt.tight_layout()

        # Test metrics
        test_metrics, x_transformed_array, y_transformed_array = test_model(
            model=model,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            evaluate=True,  # get final result(evaluate=True)
        )

        # Transformation of data through time stepping
        # Create reduced array
        # 1. Remove dim due to batches
        # 2. Transform form torch tensor to numpy array
        x_transformed_reduced, y_transformed_reduced = preprocess_transformed_array(
            x_transformed_array, y_transformed_array
        )
        # Color for reduced labels
        color_transformed_reduced = np.vectorize(color_label_dict.get)(
            y_transformed_reduced
        )

        # Save result as model privat varibles
        model.test_metrics = test_metrics
        model.x_transformed_array = x_transformed_array
        model.y_transformed_array = y_transformed_array
        model.x_transformed_reduced = x_transformed_reduced
        model.y_transformed_reduced = y_transformed_reduced
        model.color_transformed_reduced = color_transformed_reduced
        model.train_loss_per_epoch = train_loss_per_epoch
        model.test_loss_per_epoch = test_loss_per_epoch
        model.train_acc_per_epoch = train_acc_per_epoch
        model.test_acc_per_epoch = test_acc_per_epoch
        model.confusion_matrix = cm

        model_object_dict[activation_function.__name__] = model

    plt.show()
    return model_object_dict
