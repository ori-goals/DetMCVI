# DetMCVI, a scalable solver for Deterministic POMDPs

## Compilation
```sh
mkdir build && cd build
cmake ..
make
```

## Deterministic POMDPs
A deterministic POMDP (Det-POMDP) is a POMDP with a deterministic transition function and a deterministic observation function.
All of the uncertainty in the Det-POMDP is contained within the initial belief.

## DetMCVI
DetMCVI is a modification of Monte-Carlo Value Iteration (Bai et al. 2011) for **deterministic** problems.

## Example Domains
Example domain implementations and test code can be found in [experiments/README.md](experiments/README.md).

## References
Bai, H. et al. (2011) ‘Monte Carlo value iteration for continuous-state POMDPs’, in. Algorithmic Foundations of Robotics IX: Selected Contributions of the Ninth International Workshop on the Algorithmic Foundations of Robotics, Springer, pp. 175–191.
