# Global parameters


# Model Architecture:

# ------ Shallow Model Architectures -------- #
# (Shallow) Euler standard
shallow_kwargs_1 = {
    "name": "Shallow Euler Standard ODE",
    "hidden_dim": 25,  # m_s
    "hidden_internal_dim": None,  # p_s
    "num_hidden_layers": 3,  # l
    "method": "euler",
    "variant": "standard",
}

# (Shallow) Euler UT m=p
shallow_kwargs_2_1 = {
    "name": "Shallow Euler UT ODE with m=p",
    "hidden_dim": 18,  # m_u
    "hidden_internal_dim": 18,  # p_u
    "num_hidden_layers": 3,  # l
    "method": "euler",
    "variant": "UT",
}

# (Shallow) Euler UT m≠p
shallow_kwargs_2_2 = {
    "name": "Shallow Euler UT ODE with m≠p",
    "hidden_dim": 10,  # m_u
    "hidden_internal_dim": 32,  # p_u
    "num_hidden_layers": 3,  # l
    "method": "euler",
    "variant": "UT",
}

# (Shallow) RK4 standard
shallow_kwargs_3 = {
    "name": "Shallow RK4 Standard ODE",
    "hidden_dim": 12,  # m_s
    "hidden_internal_dim": None,  # p_s
    "num_hidden_layers": 3,  # l
    "method": "rk4",
    "variant": "standard",
}

# (Shallow) RK4 UT m=p
shallow_kwargs_4_1 = {
    "name": "Shallow RK4 UT ODE with m=p",
    "hidden_dim": 9,  # m_u
    "hidden_internal_dim": 9,  # p_u
    "num_hidden_layers": 3,  # l
    "method": "rk4",
    "variant": "UT",
}

# (Shallow) RK4 UT m≠p
shallow_kwargs_4_2 = {
    "name": "Shallow RK4 UT ODE with m≠p",
    "hidden_dim": 10,  # m_u
    "hidden_internal_dim": 8,  # p_u
    "num_hidden_layers": 3,  # l
    "method": "rk4",
    "variant": "UT",
}


# ------ Deep Model Architectures -------- #
# (Deep) Euler standard
deep_kwargs_1 = {
    "name": "Deep Euler Standard ODE",
    "hidden_dim": 11,  # m_s
    "hidden_internal_dim": None,  # p_s
    "num_hidden_layers": 15,  # l
    "method": "euler",
    "variant": "standard",
}

# (Deep) Euler UT m=p
deep_kwargs_2_1 = {
    "name": "Deep Euler UT ODE with m=p",
    "hidden_dim": 8,  # m_u
    "hidden_internal_dim": 8,  # p_u
    "num_hidden_layers": 15,  # l
    "method": "euler",
    "variant": "UT",
}

# (Deep) Euler UT m≠p
deep_kwargs_2_2 = {
    "name": "Deep Euler UT ODE with m≠p",
    "hidden_dim": 10,  # m_u
    "hidden_internal_dim": 6,  # p_u
    "num_hidden_layers": 15,  # l
    "method": "euler",
    "variant": "UT",
}

# (Deep) RK4 standard
deep_kwargs_3 = {
    "name": "Deep RK4 Standard ODE",
    "hidden_dim": 5,  # m_s
    "hidden_internal_dim": None,  # p_s
    "num_hidden_layers": 15,  # l
    "method": "rk4",
    "variant": "standard",
}

# (Deep) RK4 UT m=p
deep_kwargs_4_1 = {
    "name": "Deep RK4 UT ODE with m=p",
    "hidden_dim": 4,  # m_u
    "hidden_internal_dim": 4,  # p_u
    "num_hidden_layers": 15,  # l
    "method": "rk4",
    "variant": "UT",
}

# (Deep) RK4 UT m≠p
deep_kwargs_4_2 = {
    "name": "Deep RK4 UT ODE with m≠p",
    "hidden_dim": 8,  # m_u
    "hidden_internal_dim": 2,  # p_u
    "num_hidden_layers": 15,  # l
    "method": "rk4",
    "variant": "UT",
}

shallow_kwargs_dict = {
    "Shallow Euler Standard": shallow_kwargs_1,
    "Shallow Euler UT m=p": shallow_kwargs_2_1,
    "Shallow Euler UT m≠p": shallow_kwargs_2_2,
    "Shallow RK4 Standard": shallow_kwargs_3,
    "Shallow RK4 UT m=p": shallow_kwargs_4_1,
    "Shallow RK4 UT m≠p": shallow_kwargs_4_2,
}

deep_kwargs_dict = {
    "Deep Euler Standard": deep_kwargs_1,
    "Deep Euler UT m=p": deep_kwargs_2_1,
    "Deep Euler UT m≠p": deep_kwargs_2_2,
    "Deep RK4 Standard": deep_kwargs_3,
    "Deep RK4 UT m=p": deep_kwargs_4_1,
    "Deep RK4 UT m≠p": deep_kwargs_4_2,
}

# num_parameters_to_learn = write to func

# Epochs
n_epochs = 50

# Plotting
color_label_dict = {0: "blue", 1: "red"}

# Dataset
n_points = 2000  # Make 1000 samples
dataset_types = ["spiral", "circle"]  # Choose between "spiral" and "circle"
