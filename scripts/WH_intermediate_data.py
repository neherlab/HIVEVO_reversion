import trajectory
import divergence

if __name__ == '__main__':
    folder_path = "data/WH/"
    trajectory.make_intermediate_data(folder_path)
    divergence.make_intermediate_data(folder_path)
