# Stochastic Processes 2018-2019/2 @ ELTE

## First lecture notes

### Introduction

- We're studying systems, which are "non-deterministic" by name.
- BUT: Macroscopic systems are always determined by microscopic fluctuations.
{IMAGE}
- Main question: do fluctuations contain any information? -> A: Yes!! -> It's important to research them.
- We're studying those systems (mostly), where fluctuations occure around their state of equilibrium.

---

### History of Brownian-motion

- The basic idea originates from [Robert Brown](https://en.wikipedia.org/wiki/Robert_Brown_(botanist,_born_1773)), 1827
- He studied the motion of pollens in liquids at rest. He saw, that particles are moving randomly in the liquid and concluded that they're moving on their own, as all living creature does that. But: he also observes the same type of motion for non-living particles too (sand, salt, etc.) and thus cannot explain the phenomenon.
- Einstein [publishes](https://pdfs.semanticscholar.org/9c1d/91a9f0a37e578ee9a6605b224ad554ec6e86.pdf) an article about the subject in 1905 and gives a mathematical model for its physics. He shows that the motion is not originates from some intrisic nature as Brown proposed, but rather from the molecular structure of liquids and gases. The medium in question contains randomly moving particles and molecules, which also randomly pushes the pollens from all directions and that produces its movement.
- Independently from Einstein, Smoluchowski [also publishes](https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.19063261405) a paper about the Brownian-motion in 1906 and comes to the same conclusion.
- Langevin [also](https://aapt.scitation.org/doi/10.1119/1.18725) gives a description, in 1908. His work is the first, which defines the diffusion coefficient with measurable physical quantities, and thus makes it possible to measure the Avogadro constant.
- Perrin, using these articles succesfully measures the Avogadro constant in the same year, in 1908.
- These discoveries greatly strengthen the atomic theory of matter, which isn't recognized in that time.

---

### Movement of a pollen
{IMAGE}
- We're observing the position of a particle from time to time at ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20tau) intervals.
- We assume, that the direction of the displacement is independent from the previous ones.  
  
- We study the 1D random-walk of a particle in the following
  
Let ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20P%20%5Cleft%28%20x%2C%20t%20%5Cright%20%29) indicate the probability, that after ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20t) time from the starting position the particle is staying in the ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cleft%20%5B%20x%2C%20x&plus;dx%20%5Cright%20%5D) interval.  
We're search for the ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20P%20%5Cleft%28%20x%2C%20t%20&plus;%20%5Ctau%20%5Cright%20%29), which represents the probability, that the particle is staying in the ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cleft%20%5B%20x%2C%20x&plus;dx%20%5Cright%20%5D) interval, at ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20t%20&plus;%20%5Ctau) time.  
If we assume, that a particle between two time intervals is "jumping" at ![equation](https://latex.codecogs.com/svg.latex?%5CDelta) distance from its previous location, we can introduce the definition: ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20d%5CDelta), which represents the probability of the jump length of a particle, which is in the ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cleft%5B%20%5CDelta%2C%20%5CDelta%20&plus;%20d%5CDelta%20%5Cright%5D) interval.  
We have the following criterions for this system:
1. ![equation](https://latex.codecogs.com/svg.latex?%5Cint_%7B-%20%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%5C%2C%20d%20%5CDelta)
2. ![equation](https://latex.codecogs.com/svg.latex?%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20%3D%20%5CPhi%20%5Cleft%28%20-%5CDelta%20%5Cright%20%29) --> There is no preferred dircetions in the system.

There is 3 different case, where a particle could be found or not in the interval in question:
- A particle is already in the ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cleft%20%5B%20x%2C%20x&plus;dx%20%5Cright%20%5D) interval at ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20t) and stays there.
- A particle jumps out the interval between the two observations in the ![equation](https://latex.codecogs.com/svg.latex?%5Cleft%5B%20t%2C%20t%20&plus;%20%5Ctau%20%5Cright%20%5D) time interval.
- A particle jumps from ![equation](https://latex.codecogs.com/svg.latex?%5CDelta) distance into the interval.

We can summarize these cases in order as follows:  
  
![equation](https://latex.codecogs.com/svg.latex?%5Csmall%20P%20%5Cleft%28%20x%2C%20t%20&plus;%20%5Ctau%20%5Cright%20%29%20dx%20%3D%20P%20%5Cleft%28%20x%2C%20t%20%5Cright%20%29%20dx%20-%20%5Cint_%7B-%20%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20d%20%5CDelta%20%5Ccdot%20P%20%5Cleft%28%20x%2C%20t%20%5Cright%20%29%20dx%20&plus;%20%5Cint_%7B-%20%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20d%20%5CDelta%20%5Ccdot%20P%20%5Cleft%28%20x%20-%20%5CDelta%2C%20t%20%5Cright%20%29%20dx)  
  
We can divide by ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csmall%20dx) - because the intergrals are independent from ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20x) - and then re-order the equation:  
  
![equation](https://latex.codecogs.com/svg.latex?P%20%5Cleft%28%20x%2C%20t%20&plus;%20%5Ctau%20%5Cright%20%29%20%5Ccancel%7Bdx%7D%20%3D%20%5C%5C%20%3D%20P%20%5Cleft%28%20x%2C%20t%20%5Cright%20%29%20%5Ccancel%7Bdx%7D%20-%20P%20%5Cleft%28%20x%2C%20t%20%5Cright%20%29%20%5Ccdot%20%5Cunderbrace%7B%5Cint_%7B-%20%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20d%20%5CDelta%7D_%7B%5Ctext%7B%5Cnormalfont%20%3D1%20because%20of%20first%20condition%7D%7D%20%5Ccancel%7Bdx%7D%20&plus;%20%5Cint_%7B-%20%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20d%20%5CDelta%20%5Ccdot%20P%20%5Cleft%28%20x%20-%20%5CDelta%2C%20t%20%5Cright%20%29%20%5Ccancel%7Bdx%7D)  
![equation](https://latex.codecogs.com/svg.latex?P%20%5Cleft%28%20x%2C%20t%20&plus;%20%5Ctau%20%5Cright%20%29%20%3D%20%5Ccancel%7BP%20%5Cleft%28%20x%2C%20t%20%5Cright%20%29%20-%20P%20%5Cleft%28%20x%2C%20t%20%5Cright%20%29%20%5Ccdot%201%7D%20&plus;%20%5Cint_%7B-%20%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20%5Ccdot%20P%20%5Cleft%28%20x%20-%20%5CDelta%2C%20t%20%5Cright%20%29%5C%2C%20d%20%5CDelta)  
  
So we get the following equation at the end:  
  
![equation](https://latex.codecogs.com/svg.latex?P%20%5Cleft%28%20x%2C%20t%20&plus;%20%5Ctau%20%5Cright%20%29%20%3D%20%5Cint_%7B-%20%5Cinfty%7D%5E%7B%5Cinfty%7D%20%5CPhi%20%5Cleft%28%20%5CDelta%20%5Cright%20%29%20%5Ccdot%20P%20%5Cleft%28%20x%20-%20%5CDelta%2C%20t%20%5Cright%20%29%5C%2C%20d%20%5CDelta)  
  
Which we call the *Chapman-Kolmogorov-equation*.
