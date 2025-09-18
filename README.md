# DetMCVI
Official repo for **DetMCVI**: a scalable solver for deterministic POMDPs.

[ArXiv](https://arxiv.org/abs/2505.00596)

## Compilation
```sh
mkdir build && cd build
cmake ..
make
```

## Deterministic POMDPs
A deterministic POMDP (Det-POMDP) is a POMDP with a deterministic transition function and a deterministic observation function.
All of the uncertainty in the Det-POMDP is contained within the initial belief.

DetMCVI is a modification of Monte-Carlo Value Iteration (Bai et al. 2011) for **deterministic** problems.

## Example Domains
Example domain implementations and test code can be found in [experiments/README.md](experiments/README.md).

## References
Bai, H. et al. (2011) ‘Monte Carlo value iteration for continuous-state POMDPs’, in. Algorithmic Foundations of Robotics IX: Selected Contributions of the Ninth International Workshop on the Algorithmic Foundations of Robotics, Springer, pp. 175–191.

## Citation
To cite DetMCVI:
```
@inproceedings{schutz2025finitestatecontrollerbasedoffline,
      title={A Finite-State Controller Based Offline Solver for Deterministic POMDPs}, 
      author={Alex Schutz and Yang You and Matias Mattamala and Ipek Caliskanelli and Bruno Lacerda and Nick Hawes},
      year={2025},
      booktitle={IJCAI}, 
}
```
