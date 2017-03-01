import os
import subprocess

import numpy


def read_aims(path):
    elements = []
    # [(coords, unit, energy)
    steps = []

    with open(path, 'r') as f:
        start = False
        initial = False

        temp_coords = []
        temp_unit = []
        temp_energy = None

        for line in f:
            line = line.strip()
            if not line:
                continue

            if "Total energy uncorrected" in line:
                temp_energy = line.split()[5]

            if "x [A]" in line or "Unit cell:" in line:
                if len(temp_unit):
                    continue
                start = True
                continue

            if start:
                parts = line.split()
                if len(parts) < 2:
                    start = False
                    continue

                # Input
                if line.startswith('|'):
                    if "Atom" in line:
                        continue
                    if "Species" not in line:
                        temp_unit.append(parts[1:])
                        continue
                    try:
                        ele = parts[3]
                        temp_coords.append(parts[4:7])
                        elements.append(ele)
                    except IndexError:
                        start = False
                else:
                    if "lattice_vector" in line:
                        temp_unit.append(parts[1:])
                        continue
                    try:
                        temp_coords.append(parts[1:4])
                    except IndexError:
                        import pdb; pdb.set_trace()
                        start = False



            if temp_energy is not None:
                steps.append((temp_coords, temp_unit, temp_energy))
                temp_coords = []
                temp_unit = []
                temp_energy = None
    return elements, steps


def write_data(base_name, elements, steps):
    """
    xx xy xz
    yx yy yz
    zx zy zz
    E
    ele0 x0 y0 z0
    ...
    elen xn yn zn
    """
    for i, (coords, unit, energy) in enumerate(steps):
        with open(base_name + "__%04d.cry" % i, 'w') as f:
            unit_string = '\n'.join(' '.join(x) for x in unit)
            f.write(unit_string)
            f.write('\n%s\n' % energy)
            coord_string = '\n'.join(ele + ' ' + ' '.join(x) for ele, x in zip(elements, coords))
            f.write(coord_string)


def read_data_energies(path):
    energies = []
    for name in sorted(os.listdir("data")):
        path = os.path.join("data", name)
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i == 3:
                    energies.append(float(line.strip()))
                    break
    return energies


if __name__ == '__main__':
    PATH = "TCS3_SG14"
    for directory in os.listdir(PATH):
        path = os.path.join(PATH, directory, 'aims.out')
        elements, steps = read_aims(path)
        energies = [float(x[2]) for x in steps]
        energies = numpy.array(energies)
        print directory, len(energies), energies.mean(), energies.std(), energies.max() - energies.min()
        base_name = "data/%s/%04d" % (directory, len(steps)-1)
        subprocess.call(["mkdir", "-p", "data/%s" % directory])
        write_data(base_name, elements, steps)

