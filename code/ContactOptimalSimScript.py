"""
- Call `ContactOptimalSim` to verify contact optimization
created on 2025-03-11
"""
from pathlib import Path
import itertools
import yaml
import numpy as np
import subprocess
dir_path = Path(__file__).parent

def main():
    contact_angle = {11:90, 22:90, 33:90, 44:90, 55:90, 66:90, 77:90, 88:90, 99:90,
                    21:-90, 32:-90, 43:-90, 54:-90, 65:-90, 76:-90, 87:-90, 98:-90, 109:-90,
                    111:0, 112:0, 113:0, 114:0, 115:0, 116:0, 117:0, 118:0, 119:0}

    # marker = [103, 104]
    # marker = [92, 104]
    marker = [83, 93]

    deform_strain = {}

    contact_all = list(contact_angle.keys())
    contact_combs = list(itertools.combinations(contact_all, 2))
    for contact in contact_combs:
        print(f"Contact combiantion: {contact} ---------------------------")
        cmd = ["python", "ContactOptimalSim.py", "--contact"] + [str(x) for x in contact] + ["--marker"] + [str(x) for x in marker]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        full_output = result.stdout
        # print("Full output:", full_output)
        last_loss_value = None
        final_result_line = None
        for line in full_output.splitlines():
            if line.startswith("Last loss:"):
                print(line)
                try:
                    # Extract the numeric value following "Last loss:"
                    loss_str = line.split("Last loss:")[1].strip()
                    last_loss_value = float(loss_str)
                except ValueError:
                    print("Could not convert last loss value to float.")
            if line.startswith("Final deform strain:"):
                final_result_line = line
                break
        
        if last_loss_value is not None and last_loss_value > 2.e-7:
            print(f"Can't finish task.")
            continue

        if final_result_line is not None:
            # Extract the numeric value after the marker.
            try:
                output_str = final_result_line.split("Final deform strain:")[1].strip()
                output_float = float(output_str)
                deform_strain[tuple(contact)] = output_float
                print("Deformation strain (as float):", output_float)
            except ValueError:
                print("Failed to convert final result to float.")
        else:
            print("Final result marker not found in output.")

    # Save the manipulability dictionary to a YAML file
    with open(f"{dir_path}/data/deform_strain.yaml", 'w') as file:
        yaml.dump(deform_strain, file)


if __name__ == "__main__":
    main()