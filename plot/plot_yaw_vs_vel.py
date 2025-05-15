import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

controllers = ["VI", "TOVI"]
robots = ["Draco", "G1"]
simulators = ["PB", "Mujoco"]


# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]
# sns.set_theme(style="whitegrid", font="serif")


def col_name_from_params(c: str, r: str, s: str) -> str:
    return f"{c} ({r}, {s})"


def main():
    parser = argparse.ArgumentParser(description="Plot Maximum Yaw Rate vs Velocity from CSV using Seaborn.")
    parser.add_argument("filename", type=str, help="Path to the input CSV file.")
    parser.add_argument("title", type=str, help="Title of the plot")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="experiment_data/graphs/yaw_vs_vel.png",
        help="Path to save the output plot.",
    )
    args = parser.parse_args()

    # Read the CSV file
    try:
        df = pd.read_csv(args.filename)
    except FileNotFoundError:
        print(f"Error: The file '{args.filename}' was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    x_col_name = df.columns[0]

    vi_lines = [x_col_name]
    tovi_lines = [x_col_name]
    for controller in controllers:
        for robot in robots:
            names = [col_name_from_params(controller, robot, sim) for sim in simulators]
            new_name = f"{controller} ({robot})"
            df[new_name] = df[names].mean(axis=1)
            if controller == "VI":
                vi_lines.append(new_name)
            else:
                tovi_lines.append(new_name)

    df_vi = df[vi_lines].melt(id_vars=[x_col_name], var_name="Configuration", value_name="Yaw Rate")
    df_tovi = df[tovi_lines].melt(id_vars=[x_col_name], var_name="Configuration", value_name="Yaw Rate")

    # Create the plot
    sns.set_theme(style="darkgrid", font="serif", context="talk")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_tovi, x=x_col_name, y="Yaw Rate", hue="Configuration", marker="o")
    sns.lineplot(data=df_vi, x=x_col_name, y="Yaw Rate", hue="Configuration", marker="o", linestyle="--")

    # Set labels as requested
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Maximum Yaw Rate (rad/s)")
    plt.title(f"Turn Speeds for {args.title} Velocity Tracking")
    plt.tight_layout()
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
