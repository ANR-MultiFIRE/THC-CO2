import os
import shutil
import pathlib
tests_dir_ = pathlib.Path(__file__).parent.absolute()
tests_dir = str(tests_dir_)
thermo_dir_ = pathlib.Path(tests_dir).parent.absolute()
thermo_dir = str(thermo_dir_)
TH_Model_dir = str(pathlib.Path(thermo_dir_).parent.absolute())

shutil.copy(thermo_dir + '/materials/material.py', thermo_dir + '/materials/materials_constitutive_laws.py')
shutil.copy(tests_dir + '/case_input.py', TH_Model_dir + '/temp.py')
os.system('python3 ' + TH_Model_dir + '/temp.py ')
