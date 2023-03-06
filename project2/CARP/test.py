import os
import subprocess

if __name__ == '__main__':
    paths = []
    for root, dirs, files in os.walk('./CARP_samples/'):
        for f in files:
            paths.append(os.path.join(root, f))
    for filepath in paths:
        print('-------------------------' + filepath +
              '-----------------------------')
        str0 = ("python CARP_solver_ver1_0_pathScanning.py " +
                filepath + " -t 300 -s 0")
        str1 = ("CARP_solver_ver1_1_pathScanningPro.py " +
                filepath + " -t 300 -s 0")
        str2 = ("CARP_solver_ver2_1_argument_merge.py " +
                filepath + " -t 300 -s 0")
        str3 = ("CARP_solver_ver3_1_MEANS.py " + filepath + " -t 300 -s 0")
        # subprocess.Popen(str0)
        subprocess.run(str1)
        # subprocess.Popen(str2)
        # subprocess.Popen(str3)

# pathScanning:      4201, 6446, 370, 309, 212, 504, 370
# pathScanning-Pro:  3774, 5608, 316, 275, 173, 410, 290
# argument-merge:    5060, 7076, 395, 339, 197, 492, 352
# MEANS:             3589, 5272, 316, 275, 173, 418, 292
