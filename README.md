# LDS_KF_KS

Go to [Joaquin Rapela's Github repo](https://github.com/joacorapela/lds_python.git) for documentation and Python code about LDS, KF, KS.

To produce position, velocity, acceleration plots for a simulated mouse run simulation.py.

To produce position, velocity, acceleration plots for the real mouse run detect.py.

Running simulation.py
- By default medium noise simulation generated
- Option to have a highly noisy simulation by commenting out low noise variable assignments
- Includes learnParams() for learning parameters of LDS

Running detect.py
- Pass True to main() to use KF, KS on csv values, and False to use KF, KS on video
Note: The csv and video data I used do not belong to me so you will have to provide your own video (same directory as detect.py) of a moving object or csv values (time,x coordinate,y coordinate) in a folder called 'data'

