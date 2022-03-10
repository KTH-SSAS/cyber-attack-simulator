# Intro

Things to define:
- Roadmap
- Interface Specification
- Big Picture (add details, maybe docs?)


## Interface Specification

Environments: Interface between attacker and defender
It is what is giving rewards.


## High Level Diagram Notes
At this level it is reasonable to say that this is
in python. Arrows are basically python function calls.

The act/observe interface involves actually many
things:

- The Attacker/Defender has a set of available decisions
  these decisions are mapped to an action space, the simulator will
  convert an action from the action space to an operation on the instance
  model.  An operation on the instance model will be reflected on the
  attack graph.  The observation space will be different depending on
  attacker/defender.

- Changes in the instance model should be translated to changes in the
attack graph.  But there are changes on the attack graph which do not
translate to the instance model. For example "activateAV" does not remove
an asset but maybe disables some attack steps on the attack graph.

- The Observe function provides two things:
    - Observable State
    - Rewards (This is where the rewards happen)

- The arrow between Action Filter and State is double-edged
because the simulator should be able to say if the action is valid.
Validity of an action depends among other things on the language.
