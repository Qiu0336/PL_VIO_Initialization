

PL_VIO_Initialization
===============

## Description

We propose a novel visual-inertial initialization method integrating both point and line features.
Specifically, a closed-form method of line features is presented for initialization, which is combined with point-based method to
build an integrated linear system. Parameters including initial velocity, gravity, point depth and line’s endpoints depth can be
jointly solved out. Furthermore, to refine these parameters, a global optimization method is proposed, which consists of two
novel nonlinear least squares problems for respective points and lines. Both gravity magnitude and gyroscope bias are considered
in refinement. Extensive experimental results on both simulated and public datasets show that integrating point and line features
in initialization stage can achieve higher accuracy and better robustness compared with pure point-based methods.

**Paper:** Hong Liu, Junyin Qiu, Weibo Huang. **Integrating Point and Line Features for Visual-Inertial Initialization**, *International Conference on Robotics and Automation (ICRA2022). IEEE: 9470-9476. May 23-27, 2022.* **[PDF](https://github.com/Qiu0336/PL_VIO_Initialization/blob/main/paper/Integrating_Point_and_Line_Features_for_Visual-Inertial_Initialization.pdf)**.

Graph illustration of points and lines combined optimization problem.
![image](https://github.com/Qiu0336/PL_VIO_Initialization/blob/main/paper/pic/graph.png)

Diagram of simulation.
![image](https://github.com/Qiu0336/PL_VIO_Initialization/blob/main/paper/pic/simulation.png)



## License

Dashgo_slam is under [GPLv3 license](https://github.com/Qiu0336/PL_VIO_Initialization/blob/main/LICENSE).


