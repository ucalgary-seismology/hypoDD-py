# hypoDD-py
Fully function python3 version of hypoDDv1.3 (from https://www.ldeo.columbia.edu/~felixw/hypoDD.html)

Includes both the ph2dt and hypoDD commands from the original software.

Ph2dt: creates the necessary input files for hypoDD from a phase.dat format file.
hypoDD: runs double-difference relocation on catalog and outputs relocated event catalog as well as other information (such as a residual file).


To run ph2dt:

python ph2dt.py [inputfile]


To run hypoDD:

python hypoDD_run.py [inputfile]



Currently only tested on small example problems. Expect updates with more testing and application.
