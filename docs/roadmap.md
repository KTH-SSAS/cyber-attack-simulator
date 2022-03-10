# Roadmap

- Develop a procedural generator (???)
    This should generate instance models

- Defense steps (Nyberg)
    Implement the effects of enabling a defense steps.  The idea is to
    mark an attack step/instance model asset as disabled

- Human Interface (användargränssnitt) (Nyberg)
    How does a human interact with the interface?
    We need to develop an interface for a human user.
    What information is presented and how is it presented?

- Change of Simulator State
    Developing mechanisms to change the state of the instance model and
    the attack graph according to the laws of MAL

- Initial Attack Surface
    Where could an attacker start? This depends on the language

- Reward Engineering
    How should reward work? Should it be tied to specific attack steps?
    How do we designate values? Attackers may have different goals.
    For example, there may be attackers focused on availability others
    on confidentiality

- Simulation and TTC 
    Developing mechanisms to change the state of the instance model and
    the attack graph according to the laws of MAL
    
- Testing of the translator (mgg) (Nebbione)
    The translator needs more test cases and scenarios, it has still
    rough edges; Models which are characterized by complex features
    (such as a combination of let statements, override operators and
    transitive attack steps) should be tested
