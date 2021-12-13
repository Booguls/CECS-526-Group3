# CECS-526-Group3
A project oriented to simulate edge devices equipped with machine learning applications. This project aims to demonstrate a methodology to adjusut machine learning model parameters remotely.

## Code Execution steps

1. git clone command: 
$ git clone https://github.com/Booguls/CECS-526-Group3.git

2. Go into the file path of the cloned folder
$ cd CECS-526-Group3

3. Build image:
$ docker build -t mlappimage:latest .

4. See created image:
$ docker images

5. Run the container:
$ docker run  mlappimage:latest
o/p: Multi-Layer Perceptron accuracy score with default settings: 78.30733325144331

