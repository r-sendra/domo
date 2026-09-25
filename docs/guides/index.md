# Guides

How to run the library, how to add to it, and what to do when it breaks.
[Concepts](../concepts/index.md) says why it is shaped this way and the
[reference](../api/index.md) owns the signatures; these pages assume neither.

<div class="grid cards" markdown>

-   __Running and training__

    ---

    Which entry point trains which policy, what it costs in steps and
    hours, how to watch a run and resume it, and how a checkpoint becomes
    a blessed policy.

    [:octicons-arrow-right-24: Running and training](running.md)

-   __Extending DOMO__

    ---

    Nine extension points, each with the template file that already does
    it right and the test file to copy.

    [:octicons-arrow-right-24: Extending DOMO](extending.md)

-   __Troubleshooting__

    ---

    Symptom, cause, fix — Genesis quirks, checkpoint formats, budget
    surprises and the things that only bite on a headless machine.

    [:octicons-arrow-right-24: Troubleshooting](troubleshooting.md)

</div>

## Common tasks

| I need to… | Go to |
|------------|-------|
| train a gait | [Training](running.md#training) |
| evaluate a checkpoint | [Evaluation](running.md#evaluation) |
| promote a stable policy | [Promoting a checkpoint](running.md#promoting-a-checkpoint) |
| add a reward term | [Add a reward term](extending.md#add-a-reward-term) |
| add a skill card | [Add a skill and a card](extending.md#add-a-skill-and-a-card) |
| add a physics backend | [Add a physics backend](extending.md#add-a-physics-backend) |
| fix a Genesis error | [Genesis](troubleshooting.md#genesis) |
