import utilities as utilities

# Global parameters

# Epochs
n_epochs = 200

# Plotting
color_label_dict = {0: "blue", 1: "red"}

# Dataset
n_points = 2000  # Make 1000 samples
dataset_kwargs_dict = {  # Choose between "spiral" and "circle"
    "Spiral": {
        "dataset_name": "Spiral dataset",
        "get_data_function": utilities.generate_spiral_data,
        "color_label_dict": color_label_dict,
        "n_points": n_points,
        "noise": 0.8,
        "degree": 540,
        "input_dim": 2,
        "output_dim": 1,
    },
    "Circle": {
        "dataset_name": "Circle dataset",
        "get_data_function": utilities.generate_circle_data,
        "color_label_dict": color_label_dict,
        "n_points": n_points,
        "noise": 0.1,
        "factor": 0.8,
        "input_dim": 2,
        "output_dim": 1,
    },
}

# Model Architecture:
# (Shallow) Standard ODE
shallow_standard_kwargs = {
    "name": "Shallow Standard ODE",
    "hidden_dim": 25,  # m_s
    "hidden_internal_dim": None,  # p_s
    "num_hidden_layers": 3,  # l
    "variant": "standard",
}

# (Shallow) General quadratic ODE
shallow_general_quad_kwargs = {
    "name": "Shallow General ODE with m=p",
    "hidden_dim": 18,  # m_u
    "hidden_internal_dim": 18,  # p_u
    "num_hidden_layers": 3,  # l
    "variant": "general",
}

# (Shallow) General rectangular ODE
shallow_general_rec_kwargs = {
    "name": "Shallow General ODE with m≠p",
    "hidden_dim": 10,  # m_u
    "hidden_internal_dim": 31,  # p_u
    "num_hidden_layers": 3,  # l
    "variant": "general",
}

# (Deep) Standard ODE
deep_standard_kwargs = {
    "name": "Deep Standard ODE",
    "hidden_dim": 11,  # m_s
    "hidden_internal_dim": None,  # p_s
    "num_hidden_layers": 15,  # l
    "variant": "standard",
}

# (Deep) General quadratic ODE
deep_general_quad_kwargs = {
    "name": "Deep General ODE with m=p",
    "hidden_dim": 8,  # m_u
    "hidden_internal_dim": 8,  # p_u
    "num_hidden_layers": 15,  # l
    "variant": "general",
}

# (Deep) General rectangular ODE
deep_general_rec_kwargs = {
    "name": "Deep General ODE with m≠p",
    "hidden_dim": 10,  # m_u
    "hidden_internal_dim": 6,  # p_u
    "num_hidden_layers": 15,  # l
    "variant": "general",
}

ode_kwargs_dict = {
    "Shallow Standard ODE": shallow_standard_kwargs,
    "Shallow General Quadratic ODE": shallow_general_quad_kwargs,
    "Shallow General Rectangular ODE": shallow_general_rec_kwargs,
    "Deep Standard ODE": deep_standard_kwargs,
    "Deep General Quadratic ODE": deep_general_quad_kwargs,
    "Deep General Rectangular ODE": deep_general_rec_kwargs,
}
solver_method_kwargs_dict = {
    # "Simple NN": {"method": "neural"},
    "Forward Euler": {"method": "euler"},
    "Runge-Kutta 4": {"method": "rk4"},
}


# ----------------------------------------------------------------
# OLD


"""
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
"""
