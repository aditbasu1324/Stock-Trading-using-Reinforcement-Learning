"""
diagnostics.py

Standard diagnostic checklist for tabular Q-learning agents, developed across
the Blackjack Monte Carlo work and the cliff-walking / AAPL trading agent work.

Run this checklist after every training run, before trusting the result.

Usage:
    from diagnostics import run_full_diagnostics
    run_full_diagnostics(agent, visit_counts, delta_history, seed_agents,
                          greedy_actions, random_baseline, buy_hold_baseline,
                          train_final, test_final)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_train_test_summary(train_results, test_results):
    """0. TRAIN/TEST SUMMARY -- the headline mean +/- std final portfolio value
    across seeds, train and test. Printed first, ahead of all the diagnostic
    detail below, since it's the number that actually answers "did this run
    work" before digging into why.
    train_results, test_results: arrays of per-seed final portfolio values --
    the same arrays every training loop in the notebook already produces (e.g.
    train_results/test_results for the main runs, or
    best_q["train_results"]/best_q["test_results"] for a tuned run's dict).
    """
    train_results = np.asarray(train_results)
    test_results = np.asarray(test_results)
    print(f"[Train/Test] Train -- mean: {train_results.mean():.2f}  std: {train_results.std():.2f}")
    print(f"[Train/Test] Test  -- mean: {test_results.mean():.2f}  std: {test_results.std():.2f}")


def check_coverage(visit_counts):
    """1. COVERAGE -- did every (state, action) pair get enough samples?
    visit_counts: array of shape (..., n_actions) -- any number of leading
    state dimensions (e.g. (n_states, n_actions) or (n_bins, n_cash_bins,
    n_actions)) is fine -- incremented every time agent.update(...) is
    called for that (state, action) pair.
    """
    n_unvisited = np.sum(visit_counts == 0)
    total = visit_counts.size
    print(f"[Coverage] Unvisited (state, action) pairs: {n_unvisited} / {total} "
          f"({100*n_unvisited/total:.1f}%)")
    if n_unvisited > 0:
        idx = list(zip(*np.where(visit_counts == 0)))  # works for any number of dimensions
        print(f"  Unvisited pairs (state..., action): {idx[:10]}"
              f"{' ...' if n_unvisited > 10 else ''}")
    return n_unvisited


def check_convergence(delta_history, threshold=1e-3):
    """2. CONVERGENCE -- has the Q-table actually stabilized?
    delta_history: list/array of max|Q_new - Q_old| -- historically recorded
    once per episode (whole-table diff); plot_convergence now instead tracks
    this per training step, aggregated across seeds (see its docstring). This
    function itself just reads off delta_history[-1], so it works with either,
    but "final delta" means "the last training step" if given the newer arrays.
    """
    final_delta = delta_history[-1]
    converged = final_delta < threshold
    print(f"[Convergence] Final delta: {final_delta:.6f} "
          f"({'CONVERGED' if converged else 'NOT YET CONVERGED, consider more episodes'})")
    return converged


def _epsilon_floor_episode(epsilon_start, epsilon_decay, epsilon_floor, n_episodes):
    """First episode (1-indexed, matching the convergence plot's x-axis) at which
    epsilon = max(epsilon * epsilon_decay, epsilon_floor) settles at the floor.

    This is fully deterministic: the epsilon schedule never depends on the random
    seed or on anything the agent does, only on these three numbers and the
    episode count. So there's no need to measure it empirically inside a training
    loop for some particular seed -- every seed follows the exact same epsilon
    trajectory, and it's cheaper and exact to just simulate the tiny recurrence
    directly here.

    Returns None if the floor is never reached within n_episodes (e.g. epsilon_floor
    is 0 and epsilon only asymptotically approaches it, or epsilon_decay == 1.0).
    """
    eps = epsilon_start
    for ep in range(1, n_episodes + 1):
        eps = max(eps * epsilon_decay, epsilon_floor)
        if eps <= epsilon_floor:
            return ep
    return None


def plot_convergence(delta_history_mean, delta_history_max,
                      epsilon_start=None, epsilon_decay=None, epsilon_floor=None,
                      policy_flip_history=None, episode_reward_history=None,
                      policy_flip_history_b=None):
    """2b. CONVERGENCE PLOT -- visualize how much the Q-table actually moves,
    PER TRAINING STEP (one Q-learning update = one timestep of one episode),
    not per whole episode. Each `agent.update(...)` call only ever touches a
    single (state, action) cell, so at this granularity there's no "whole
    table" left to take a mean/max over the way there was at episode
    granularity -- the only remaining source of variation at a fixed step is
    which SEED you're looking at. So here, mean/max are computed ACROSS SEEDS
    at each step: "Mean delta" is that step's update size averaged over every
    seed; "Max delta" is the single largest update any seed made at that step.
    delta_history_mean, delta_history_max: lists of the same length, one
    entry per TRAINING STEP -- i.e. length episodes * (timesteps per episode),
    every episode's steps concatenated back-to-back in training order. This
    will be a much longer, denser plot than one point per episode.
    epsilon_start/epsilon_decay/epsilon_floor: optional. If all three are given
    AND at least one of policy_flip_history/episode_reward_history is also
    given (needed to know how many steps make up one episode), a vertical line
    is drawn at the step where epsilon first settles at the floor (see
    _epsilon_floor_episode) -- lets you see whether Q-value convergence lines
    up with epsilon having stopped decaying (agent mostly exploiting from then
    on) or happens independently of it. Leave as None to skip the line.
    policy_flip_history: optional, one entry per EPISODE (not training step) --
    the fraction of states whose greedy action (argmax Q) changed since the
    previous episode, already averaged across seeds. This is a different
    notion of convergence than the delta panels: |Q_new - Q_old| can stay
    noticeably nonzero forever (Q keeps refining its value estimates) even
    once the actual POLICY has completely settled, since a state's argmax
    doesn't care how much the runner-up action's value moves, only whether
    the ranking flips. When given, adds a panel plotting this per episode.
    policy_flip_history_b: optional, double Q-learning only -- same shape and
    meaning as policy_flip_history, but for a SECOND table's own greedy policy
    (e.g. argmax(Q_b) flips) tracked independently from the first
    (policy_flip_history, e.g. argmax(Q_a) flips). Q_a and Q_b are each only
    updated on roughly half of the training steps (the coin-flip in
    Agent_double_q.update), so they can settle at different rates -- plotting
    the combined table's flips alone would hide that. Only used when
    policy_flip_history is also given: both are then drawn as two labeled,
    overlaid lines on the SAME single panel (still titled "Fraction of states
    with a flipped greedy action") rather than getting a panel each, so the
    two tables' convergence speed is directly comparable at a glance. Ignored
    (no effect, no error) if policy_flip_history is None. Leave as None for
    single-table agents (Q-learning, Expected SARSA) -- panel then shows just
    the one line, exactly as before this parameter was added.
    episode_reward_history: optional, one entry per EPISODE -- the TOTAL
    reward accumulated during that whole training episode (summed over every
    timestep in that pass through the training data, not the post-hoc greedy
    evaluation reported as train_final/test_final), already averaged across
    seeds. This is the standard "training reward curve": since the behaviour
    policy is epsilon-greedy (not greedy), it reflects how well the agent is
    actually doing WHILE learning/exploring. When given, adds a panel
    plotting this per episode.
    """
    n_steps = len(delta_history_mean)
    step_x = range(1, n_steps + 1)

    n_panels = 2 + (policy_flip_history is not None) + (episode_reward_history is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))

    axes[0].plot(step_x, delta_history_mean)
    axes[0].set_title("Mean |Q change| per training step (across seeds)")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Mean delta across seeds")

    axes[1].plot(step_x, delta_history_max)
    axes[1].set_title("Max |Q change| per training step (across seeds)")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Max delta across seeds")

    # whichever episode-indexed array is available also tells us how many training
    # steps make up one episode (n_steps / n_episodes), needed to place the
    # epsilon-floor line correctly on the step-indexed delta panels below
    n_episodes = None
    next_panel = 2
    if policy_flip_history is not None:
        n_episodes = len(policy_flip_history)
        episode_x = range(1, n_episodes + 1)
        if policy_flip_history_b is not None:
            axes[next_panel].plot(episode_x, policy_flip_history, label="Q_a")
            axes[next_panel].plot(episode_x, policy_flip_history_b, label="Q_b")
            axes[next_panel].legend()
        else:
            axes[next_panel].plot(episode_x, policy_flip_history)
        axes[next_panel].set_title("Fraction of states with a flipped greedy action")
        axes[next_panel].set_xlabel("Episode")
        axes[next_panel].set_ylabel("Fraction flipped")
        next_panel += 1

    if episode_reward_history is not None:
        n_episodes = len(episode_reward_history)
        episode_x = range(1, n_episodes + 1)
        axes[next_panel].plot(episode_x, episode_reward_history)
        axes[next_panel].set_title("Total training reward per episode")
        axes[next_panel].set_xlabel("Episode")
        axes[next_panel].set_ylabel("Summed reward (epsilon-greedy behaviour)")
        next_panel += 1

    if epsilon_start is not None and epsilon_decay is not None and epsilon_floor is not None and n_episodes is not None:
        floor_episode = _epsilon_floor_episode(epsilon_start, epsilon_decay, epsilon_floor, n_episodes)
        if floor_episode is not None:
            steps_per_episode = n_steps / n_episodes
            floor_step = floor_episode * steps_per_episode
            for i, ax in enumerate(axes):
                x_value = floor_step if i < 2 else floor_episode  # first 2 panels are step-indexed, rest episode-indexed
                ax.axvline(x_value, color="red", linestyle="--", alpha=0.7,
                           label=f"epsilon hits floor (ep {floor_episode})")
                ax.legend()

    plt.tight_layout()
    plt.show()


def plot_episode_trajectory(first_episode_reward_trace, last_episode_reward_trace):
    """2c. WITHIN-EPISODE TRAJECTORY -- plot_convergence's panels are all one
    point PER EPISODE (~100 points, one per pass through the training data).
    This zooms into the OPPOSITE axis: within a single episode, one point PER
    TIMESTEP (~800 points, one per training day), so you can see the shape of
    how reward accrues over the course of a single pass through the data --
    not just its final total.

    Plots the CUMULATIVE reward (running sum, so this reads like an equity
    curve -- portfolio value gained so far within that one episode) against
    timestep, for the FIRST training episode (near-random behaviour, epsilon
    close to its start value) and the LAST training episode (near-final
    policy, epsilon at or close to its floor), overlaid on the same axes so
    the shift in behaviour from the start to the end of training is visible
    directly, rather than needing to reason about ~100 separate per-episode
    numbers.

    first_episode_reward_trace, last_episode_reward_trace: arrays of raw
    per-timestep reward (not already cumulative) from episode 0 and the final
    training episode respectively, already averaged across seeds -- same
    length as each other, since every episode is one full pass over the same
    training data regardless of seed or how far along training is.
    """
    first_cumulative = np.cumsum(first_episode_reward_trace)
    last_cumulative = np.cumsum(last_episode_reward_trace)
    timesteps = range(1, len(first_cumulative) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, first_cumulative, label="First episode (mostly random)")
    ax.plot(timesteps, last_cumulative, label="Last episode (near-final policy)")
    ax.set_title("Cumulative reward within one episode: start of training vs. end")
    ax.set_xlabel("Timestep within episode (training day)")
    ax.set_ylabel("Cumulative reward")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_value_trend(mean_history, max_history, value_name="Value", extra_note=""):
    """2d/2e. VALUE MAGNITUDE OVER TRAINING -- generic mean/max-per-episode plot,
    shared by plot_q_value_trend and plot_reward_trend (see their docstrings for
    what each is actually checking). Every other convergence check in this
    module tracks how much something is CHANGING (deltas, policy flips); this
    one instead tracks the actual VALUES, to answer a different question: is
    the quantity settling into a stable, sensible range, or drifting off to
    ever-larger (or ever-smaller/more negative) magnitudes? Small per-step
    deltas can still add up to steady, unbounded drift overall if they're
    consistently one-directional -- delta-based checks alone, which only ever
    look at one step at a time, wouldn't catch that.

    mean_history, max_history: lists of the same length, one entry per episode,
    already averaged across seeds.
    value_name: label used in the titles/axis labels (e.g. "Q-value", "reward").
    extra_note: optional short string appended to both titles for context.
    """
    episodes = range(1, len(mean_history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    suffix = f" ({extra_note})" if extra_note else ""

    axes[0].plot(episodes, mean_history)
    axes[0].set_title(f"Mean {value_name} per episode{suffix}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel(f"Mean {value_name} (avg across seeds)")

    axes[1].plot(episodes, max_history)
    axes[1].set_title(f"Max {value_name} per episode{suffix}")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel(f"Max {value_name} (avg across seeds)")

    plt.tight_layout()
    plt.show()


def plot_q_value_trend(q_mean_history, q_max_history):
    """2d. Q-VALUE MAGNITUDE OVER TRAINING -- is Q settling into a stable range
    or drifting unboundedly? See plot_value_trend's docstring for the general
    reasoning. q_mean_history/q_max_history: mean and max across the WHOLE
    Q-table (every state and action, not broken down by action) at the end of
    each episode, already averaged across seeds.
    """
    plot_value_trend(q_mean_history, q_max_history, value_name="Q-value")


def plot_reward_trend(reward_mean_history, reward_max_history):
    """2e. RAW REWARD MAGNITUDE OVER TRAINING -- unlike Q (a learned estimate),
    reward is the direct environment signal each step (portfolio_after -
    portfolio_before) -- it doesn't depend on Q at all, so this is an
    independent sanity check: are actual per-step rewards trending up or down
    as training progresses, separate from whatever Q currently believes.
    reward_mean_history/reward_max_history: mean and max reward received
    per step within each episode, already averaged across seeds. (Mean per
    step, not the episode's total -- see episode_reward_history/
    plot_convergence for the total-per-episode view instead.)
    """
    plot_value_trend(reward_mean_history, reward_max_history, value_name="reward",
                      extra_note="per step, epsilon-greedy behaviour")


def plot_q_gap_trend(q_gap_mean_history, q_gap_max_history):
    """2f. Q_A vs Q_B GAP OVER TRAINING (double Q-learning only) -- how far
    apart are the two tables' value estimates, and is that gap shrinking as
    training progresses? check_double_q_agreement reports this same
    |Q_a - Q_b| quantity, but only as a single post-hoc snapshot at the end
    of training; this instead tracks it every episode, the same way
    plot_q_value_trend tracks raw Q magnitude over time, so you can see
    whether the two tables are actually converging toward each other or
    staying persistently split (which would suggest eta is too high for the
    two tables to agree, or that training hasn't run long enough).
    q_gap_mean_history/q_gap_max_history: mean and max of |Q_a - Q_b| across
    the WHOLE Q-table (every state and action) at the end of each episode,
    already averaged across seeds -- same bookkeeping as q_mean_history/
    q_max_history, just computed from the gap instead of from agent.Q itself.
    """
    plot_value_trend(q_gap_mean_history, q_gap_max_history, value_name="|Q_a - Q_b|",
                      extra_note="double Q-learning table gap")


def check_policy_stability(seed_agents, min_visits=100):
    """3. POLICY STABILITY -- across independently-trained seeds, how much do
    they agree on the greedy policy at each state?

    Mechanics: for every state, take each seed's own argmax(Q) action there.
    Across the N seeds, whichever action got picked by the most seeds is that
    state's MAJORITY action, and the state's "agreement" score is
    (seeds that picked the majority) / (total seeds) -- e.g. if 26 of 50
    seeds pick the same action at a state, that state scores 26/50 = 52%.
    There's no principled reason to treat any one seed as ground truth, which
    is why agreement is measured against this per-state majority vote rather
    than an arbitrary reference seed.

    Filtering: only states visited at least `min_visits` times ON AVERAGE
    ACROSS SEEDS (summed over actions) are scored, since a barely-visited
    state's greedy action is close to arbitrary noise. Averaging (rather
    than reading off one seed, or summing across all of them) keeps the
    threshold meaning "roughly this many visits per seed" no matter how many
    seeds are passed in.

    Two numbers get printed, and they operate on DIFFERENT axes -- easy to
    mix up:
    - "States checked: X/Y" -- Y is the TOTAL NUMBER OF STATES IN THE STATE
      SPACE (n_bins * n_cash_bins), NOT the number of seeds. X of those Y
      states passed the min_visits filter above. If Y happens to equal the
      seed count elsewhere in the notebook, that's a coincidence of the
      chosen hyperparameters, not a relationship.
    - "mean/min/max agreement" -- these are computed ACROSS THE X CHECKED
      STATES (not across seeds): the average agreement score, the
      worst-agreed-upon state, and the best-agreed-upon state among the ones
      that passed the filter.

    seed_agents: list of trained agent objects (each with .Q and .visit_counts)
    min_visits: visit-count threshold described above.
    """
    n_seeds = len(seed_agents)
    n_actions = seed_agents[0].Q.shape[-1]
    policies = np.array([np.argmax(agent.Q, axis=-1) for agent in seed_agents])  # (n_seeds, *state_shape)

    avg_visits_per_state = np.mean([agent.visit_counts for agent in seed_agents], axis=0).sum(axis=-1)
    well_sampled = avg_visits_per_state >= min_visits
    n_checked = int(well_sampled.sum())
    total_states = well_sampled.size

    if n_checked == 0:
        print(f"[Policy stability] No states had >= {min_visits} average visits across seeds -- lower min_visits.")
        return None

    policies_checked = policies[:, well_sampled]  # (n_seeds, n_checked)

    majority_agreement = np.zeros(n_checked)  # fraction of seeds agreeing with the majority, per state
    for i in range(n_checked):
        counts = np.bincount(policies_checked[:, i], minlength=n_actions)
        majority_agreement[i] = counts.max() / n_seeds

    print(f"[Policy stability] States checked: {n_checked}/{total_states} total states in the state "
          f"space had >= {min_visits} average visits across seeds")
    print(f"[Policy stability] Per-state majority agreement, across those {n_checked} states -- "
          f"mean: {majority_agreement.mean()*100:.1f}%  "
          f"min: {majority_agreement.min()*100:.1f}%  max: {majority_agreement.max()*100:.1f}%")
    return majority_agreement.mean()


def check_state_action_summary(seed_agents, min_visits=100):
    """EXTRA -- per-state table of what the trained policy actually converged to:
    the majority action across seeds, what fraction of seeds agree on it, and that
    action's mean Q-value across seeds. The Q-value is the agent's own learned
    estimate of expected future return from that state under that action (what
    the "Q" in Q-learning stands for) -- NOT a single observed one-step reward.

    Same well-sampled-state filter and majority-vote reasoning as
    check_policy_stability (only states with >= min_visits average visits across
    seeds are shown, and "majority" is scored against the per-state vote across
    seeds, not an arbitrary reference seed).

    seed_agents: list of trained agent objects (each with .Q and .visit_counts).
    min_visits: visit-count threshold, same meaning as check_policy_stability.

    Prints and returns a pandas DataFrame, one row per well-sampled state, sorted
    by visits (most-visited first).
    """
    n_seeds = len(seed_agents)
    n_actions = seed_agents[0].Q.shape[-1]
    all_Q = np.array([agent.Q for agent in seed_agents])   # (n_seeds, *state_shape, n_actions)
    policies = np.argmax(all_Q, axis=-1)                   # (n_seeds, *state_shape)

    avg_visits_per_state = np.mean([agent.visit_counts for agent in seed_agents], axis=0).sum(axis=-1)
    well_sampled = avg_visits_per_state >= min_visits
    total_states = well_sampled.size
    state_indices = list(zip(*np.where(well_sampled)))  # works for any number of state dimensions

    if not state_indices:
        print(f"[State/action summary] No states had >= {min_visits} average visits across seeds -- lower min_visits.")
        return None

    rows = []
    for s in state_indices:
        counts = np.bincount(policies[(slice(None),) + s], minlength=n_actions)
        majority_action = int(np.argmax(counts))
        rows.append({
            "state": s,
            "avg_visits": round(float(avg_visits_per_state[s]), 1),
            "majority_action": majority_action,
            "agreement_pct": round(100 * counts.max() / n_seeds, 1),
            "mean_Q_majority_action": round(float(all_Q[(slice(None),) + s + (majority_action,)].mean()), 4),
        })

    df = pd.DataFrame(rows).sort_values("avg_visits", ascending=False).reset_index(drop=True)
    print(f"[State/action summary] {len(df)}/{total_states} total states had >= {min_visits} average visits across seeds")
    print(df.to_string(index=False))
    return df


def check_action_distribution(all_greedy_actions, n_actions):
    """6. ACTION DISTRIBUTION -- averaged across seeds, how often does the
    greedy policy pick each action?

    Input shape: all_greedy_actions has one entry per seed; each entry is
    that seed's own raw sequence of greedy actions over the test period
    (e.g. 50 seeds, each a list of ~251 day-by-day actions). For each seed,
    np.bincount(actions, minlength=n_actions) / len(actions) turns its raw
    counts into a length-n_actions fraction vector that sums to 1. Stacking
    all seeds gives `fractions`, shape (n_seeds, n_actions) -- rows are
    seeds, columns are actions.

    Why axis=0: mean/min/max all reduce over axis=0, which collapses the
    SEED axis and keeps the ACTION axis -- i.e. one result per action,
    aggregated over every seed. axis=1 would do the opposite (collapse
    actions per seed) and is the wrong axis here: for the mean specifically
    it's a tautology (each seed's own fractions already sum to 1, so its
    mean across actions is always exactly 1/n_actions, telling you nothing
    about behaviour); for min/max it would answer a different question (one
    seed's own least/most-used action) instead of the one this check is
    for (how much a given action's usage varies ACROSS DIFFERENT seeds).

    Note on the >95% case below: a dominant action isn't automatically a
    problem. It could mean the state isn't influencing decisions, but it
    could equally mean the agent converged on a genuinely dominant strategy
    (e.g. behaving like a simple buy-and-hold policy, which can be perfectly
    sensible for this problem) -- this check flags it, it doesn't judge it.

    all_greedy_actions: list of per-seed action sequences (each seed's own
    list/array of actions from its greedy rollout).
    """
    fractions = np.array([
        np.bincount(actions, minlength=n_actions) / len(actions)
        for actions in all_greedy_actions
    ])  # (n_seeds, n_actions)
    mean_fractions = fractions.mean(axis=0)

    print(f"[Action distribution] Mean action fractions across {len(all_greedy_actions)} seeds "
          f"(one number per action, averaged over all seeds): "
          f"{', '.join(f'{100*f:.1f}%' for f in mean_fractions)}")
    print(f"[Action distribution] Per-action spread across seeds -- min: "
          f"{', '.join(f'{100*f:.1f}%' for f in fractions.min(axis=0))}  "
          f"max: {', '.join(f'{100*f:.1f}%' for f in fractions.max(axis=0))}")
    if np.max(mean_fractions) > 0.95:
        print("  Note: one action accounts for >95% of decisions on average across seeds. "
              "Not automatically bad -- see check_seed_degeneracy for whether this is a few "
              "individual seeds or the whole population, and the docstring above for context.")
    return mean_fractions


def check_seed_degeneracy(all_greedy_actions, n_actions, threshold=0.95):
    """EXTRA -- how many INDIVIDUAL seeds have a degenerate greedy policy (one
    action dominating almost all of that seed's own decisions)?

    check_action_distribution only reports the POPULATION AVERAGE across all
    seeds, which can hide this entirely: a handful of individually degenerate
    seeds get diluted by the rest once everything is averaged together, so
    the mean can look perfectly balanced even when several seeds each
    basically always pick one action. This check looks at each seed's own
    fraction vector directly, independent of the others.

    Important: a seed being "degenerate" here is not automatically bad. In
    this trading problem it can mean the agent correctly learned that a
    simple, mostly-one-action strategy (e.g. buy once and hold) outperforms
    more active switching -- that's a legitimate, possibly GOOD outcome, not
    a failure to learn. This check reports the fact; it doesn't judge it.

    Also worth knowing: if two DIFFERENT actions both show up with a near-0%
    minimum in check_action_distribution's spread, that does not necessarily
    mean two different degenerate seeds -- a single seed that is ~100% one
    action will simultaneously show up as a near-0% minimum on both of the
    other two action columns. This check disambiguates that, since it looks
    at seeds individually rather than reading the two minimums separately.

    all_greedy_actions: list of per-seed action sequences (each seed's own
    list/array of actions from its greedy rollout).
    threshold: an action's fraction must reach this, for that one seed, for
    the seed to be counted as degenerate.
    """
    fractions = np.array([
        np.bincount(actions, minlength=n_actions) / len(actions)
        for actions in all_greedy_actions
    ])  # (n_seeds, n_actions)

    dominant_action = np.argmax(fractions, axis=1)   # each seed's own most-used action
    dominant_fraction = fractions.max(axis=1)         # how much of the time that action was used
    degenerate = dominant_fraction >= threshold
    n_degenerate = int(degenerate.sum())

    print(f"[Seed degeneracy] Seeds with one action >= {threshold*100:.0f}% of their own decisions: "
          f"{n_degenerate}/{len(all_greedy_actions)}")
    if n_degenerate > 0:
        action_counts = np.bincount(dominant_action[degenerate], minlength=n_actions)
        print(f"  Which action dominates those seeds (count per action index): {action_counts.tolist()}")
    return n_degenerate


def check_q_gaps(Q, visit_counts=None, top_n=10):
    """7. Q-VALUE GAP PER STATE -- how close is the best-vs-second-best action
    at each state? Small gaps = untrustworthy / noise-sensitive states.
    Q: array of shape (..., n_actions) -- any number of leading state
    dimensions is fine (e.g. (n_states, n_actions) or (n_bins, n_cash_bins,
    n_actions)); states are ranked across the flattened state grid.
    visit_counts (optional): same leading shape as Q. If given, also reports
    gaps for the most-visited states -- the ones whose Q-values are actually
    backed by enough data to be trusted.
    """
    sorted_q = np.sort(Q, axis=-1)
    gaps = sorted_q[..., -1] - sorted_q[..., -2]  # best minus second-best, per state (state-dims only)

    order = np.argsort(gaps, axis=None)  # flat ranking across the whole state grid
    print(f"[Q-value gaps] {top_n} closest (most borderline) states:")
    for flat_idx in order[:top_n]:
        s = np.unravel_index(flat_idx, gaps.shape)
        print(f"  state {s}: gap={gaps[s]:.4f}  Q-row={np.round(Q[s], 3)}")

    if visit_counts is not None:
        visits_per_state = visit_counts.sum(axis=-1)
        visit_order = np.argsort(visits_per_state, axis=None)[::-1]
        print(f"[Q-value gaps] {top_n} most-visited states:")
        for flat_idx in visit_order[:top_n]:
            s = np.unravel_index(flat_idx, visits_per_state.shape)
            print(f"  state {s}: visits={int(visits_per_state[s])}  gap={gaps[s]:.4f}  Q-row={np.round(Q[s], 3)}")

    return gaps


def check_baselines(train_final, test_final, random_baseline, buy_hold_baseline=None):
    """4/5. GREEDY VS BASELINES + TRAIN/TEST GAP -- the headline comparison.
    buy_hold_baseline is optional -- pass None to compare against random only.
    """
    train_r, test_r = random_baseline
    train_line = f"[Baselines] Train -- Q-learning: {train_final:.2f}  Random: {train_r:.2f}"
    test_line = f"[Baselines] Test  -- Q-learning: {test_final:.2f}  Random: {test_r:.2f}"
    if buy_hold_baseline is not None:
        train_bh, test_bh = buy_hold_baseline
        train_line += f"  Buy&Hold: {train_bh:.2f}"
        test_line += f"  Buy&Hold: {test_bh:.2f}"
    print(train_line)
    print(test_line)
    print(f"[Train/test gap] {train_final - test_final:.2f} "
          f"({'LARGE -- possible overfitting to training path' if abs(train_final-test_final) > 0.3*train_final else 'reasonable'})")
    beats_random_test = test_final > test_r
    baseline_line = f"[Baselines] Beats random on test: {beats_random_test}"
    if buy_hold_baseline is not None:
        beats_bh_test = test_final > test_bh
        baseline_line += f"   Beats buy&hold on test: {beats_bh_test}"
    print(baseline_line)


def check_double_q_agreement(seed_agents, min_visits=100):
    """EXTRA (double Q-learning only) -- how much do the two tables actually
    disagree, averaged across independently-trained seeds? Only meaningful
    for agents with separate Q_a/Q_b tables (e.g. Agent_double_q) -- every
    other diagnostic in this module only ever sees the averaged agent.Q,
    which hides this entirely.
    For each seed, restricts to states that seed's own agent visited at
    least min_visits times (summed across actions), since a barely-visited
    state's Q_a/Q_b values (and their argmax) are close to arbitrary --
    same reasoning as check_policy_stability, applied per-seed here since
    each seed is scored independently rather than against a shared reference.
    seed_agents: list of trained agents, each with .Q_a, .Q_b, .visit_counts
    """
    valid_agents = [a for a in seed_agents if hasattr(a, "Q_a") and hasattr(a, "Q_b")]
    if not valid_agents:
        print("[Double Q agreement] SKIPPED -- no agents with separate Q_a/Q_b tables")
        return None

    mean_abs_diffs = []
    action_agreements = []
    for agent in valid_agents:
        visits_per_state = agent.visit_counts.sum(axis=-1)
        well_sampled = visits_per_state >= min_visits
        if not np.any(well_sampled):
            continue
        diff = np.abs(agent.Q_a - agent.Q_b)[well_sampled]
        greedy_a = np.argmax(agent.Q_a, axis=-1)[well_sampled]
        greedy_b = np.argmax(agent.Q_b, axis=-1)[well_sampled]
        mean_abs_diffs.append(diff.mean())
        action_agreements.append(np.mean(greedy_a == greedy_b))

    if not action_agreements:
        print(f"[Double Q agreement] No seed had any states with >= {min_visits} visits -- lower min_visits.")
        return None

    mean_abs_diffs = np.array(mean_abs_diffs)
    action_agreements = np.array(action_agreements)

    print(f"[Double Q agreement] Seeds scored (had states with >= {min_visits} visits): "
          f"{len(action_agreements)}/{len(seed_agents)}")
    print(f"[Double Q agreement] Mean |Q_a - Q_b| on well-sampled states -- mean: {mean_abs_diffs.mean():.4f}  "
          f"min: {mean_abs_diffs.min():.4f}  max: {mean_abs_diffs.max():.4f}")
    print(f"[Double Q agreement] Q_a vs Q_b greedy-action agreement -- mean: {action_agreements.mean()*100:.1f}%  "
          f"min: {action_agreements.min()*100:.1f}%  max: {action_agreements.max()*100:.1f}%")
    return action_agreements.mean()


def check_q_value_changes(Q_start, Q_end, top_n=10):
    """EXTRA -- what actually changed in the Q-table over training? Breaks
    the total movement down into gains (Q went up) and losses (Q went down),
    and shows the biggest movers in each direction, rather than just the
    single max-delta number check_convergence reports.
    Q_start, Q_end: two Q-table snapshots of the same shape (e.g. agent.Q
    right after the agent is created, and agent.Q after the last episode).
    """
    diff = Q_end - Q_start
    total_gain = diff[diff > 0].sum()
    total_loss = -diff[diff < 0].sum()
    print(f"[Q-value change] Total gained: {total_gain:.4f}   Total lost: {total_loss:.4f}   "
          f"Net: {diff.sum():.4f}")

    gain_order = np.argsort(diff, axis=None)[::-1]  # biggest increases first
    print(f"[Q-value change] Top {top_n} gains:")
    shown = 0
    for flat_idx in gain_order:
        if diff.flat[flat_idx] <= 0 or shown >= top_n:
            break
        s = np.unravel_index(flat_idx, diff.shape)
        print(f"  {s}: {Q_start[s]:.4f} -> {Q_end[s]:.4f}  (+{diff[s]:.4f})")
        shown += 1

    loss_order = np.argsort(diff, axis=None)  # biggest decreases first
    print(f"[Q-value change] Top {top_n} losses:")
    shown = 0
    for flat_idx in loss_order:
        if diff.flat[flat_idx] >= 0 or shown >= top_n:
            break
        s = np.unravel_index(flat_idx, diff.shape)
        print(f"  {s}: {Q_start[s]:.4f} -> {Q_end[s]:.4f}  ({diff[s]:.4f})")
        shown += 1

    return total_gain, total_loss


def run_full_diagnostics(visit_counts=None, delta_history=None, delta_history_mean=None, seed_agents=None,
                          greedy_actions=None, n_actions=None, Q=None,
                          train_final=None, test_final=None,
                          random_baseline=None, buy_hold_baseline=None,
                          epsilon_start=None, epsilon_decay=None, epsilon_floor=None,
                          train_results=None, test_results=None, policy_flip_history=None,
                          episode_reward_history=None, first_episode_reward_trace=None,
                          last_episode_reward_trace=None, q_mean_history=None, q_max_history=None,
                          reward_mean_history=None, reward_max_history=None,
                          policy_flip_history_b=None, q_gap_mean_history=None, q_gap_max_history=None):
    """Run every diagnostic that has the inputs available. Pass None for
    anything you don't have yet -- that check is skipped with a note.

    Dormant (accepted but not called from here, kept for possible future use):
    check_convergence's printed "Final delta" line (the plot from plot_convergence was judged
    sufficient on its own), check_action_distribution / check_seed_degeneracy (judged less
    informative here since hold is frequently mechanically equivalent to sell or buy once the
    agent is fully in cash or fully invested -- see State_basic.reward), check_q_gaps, and
    check_baselines. greedy_actions/n_actions/Q/train_final/test_final/random_baseline/
    buy_hold_baseline are accepted for backward compatibility but currently unused.

    epsilon_start/epsilon_decay/epsilon_floor: optional, passed straight through to
    plot_convergence to draw a vertical line at the episode epsilon hits its floor.
    Pass the exact values used for the training run behind delta_history/delta_history_mean
    -- these vary per call site (e.g. some helper functions here default epsilon_floor to
    0.0 regardless of the notebook's global epsilon_floor), so don't assume the global.
    Leave as None (default) to skip the line, same as before this was added.

    train_results/test_results: optional arrays of per-seed final portfolio values --
    printed first (see check_train_test_summary) so the headline mean/std is visible
    before the rest of the checklist. Unrelated to the older train_final/test_final
    (a single already-reduced value, kept for the still-dormant check_baselines).

    policy_flip_history: optional, passed straight through to plot_convergence for
    its third panel (fraction of states whose greedy action changed since the
    previous episode, averaged across seeds). Leave as None to skip that panel.

    episode_reward_history: optional, passed straight through to plot_convergence
    for a panel showing total (epsilon-greedy) training reward per episode.

    first_episode_reward_trace/last_episode_reward_trace: optional. If BOTH are
    given, calls plot_episode_trajectory to show cumulative reward across a single
    episode's timesteps, comparing the first training episode against the last.

    q_mean_history/q_max_history: optional. If BOTH are given, calls
    plot_q_value_trend to show the actual Q-VALUE magnitude (not its change) over
    training -- whether Q is settling into a stable range or drifting unboundedly.

    reward_mean_history/reward_max_history: optional. If BOTH are given, calls
    plot_reward_trend to show the actual per-step REWARD magnitude (the direct
    environment signal, not a Q estimate) over training.

    policy_flip_history_b: optional, double Q-learning only -- passed straight
    through to plot_convergence. If given ALONGSIDE policy_flip_history, the
    flip-fraction panel overlays both tables' own greedy-policy flips (Q_a vs
    Q_b) as two labeled lines instead of one. No effect if policy_flip_history
    itself is None.

    q_gap_mean_history/q_gap_max_history: optional, double Q-learning only. If
    BOTH are given, calls plot_q_gap_trend to show mean/max |Q_a - Q_b| over
    training -- whether the two tables are converging toward each other.
    Independent of q_mean_history/q_max_history, which (for double Q-learning)
    should already be computed from the COMBINED agent.Q = (Q_a + Q_b) / 2,
    same as every other agent type -- this pair looks at the gap between the
    two tables instead of their combined magnitude.
    """
    print("=" * 60)
    print("DIAGNOSTIC CHECKLIST")
    print("=" * 60)

    if train_results is not None and test_results is not None:
        check_train_test_summary(train_results, test_results)
    else:
        print("[Train/Test] SKIPPED -- pass train_results and test_results to enable")
    print()

    if visit_counts is not None:
        check_coverage(visit_counts)
    else:
        print("[Coverage] SKIPPED -- pass visit_counts to enable")
    print()

    if delta_history is not None:
        if delta_history_mean is not None:
            plot_convergence(delta_history_mean, delta_history,
                              epsilon_start=epsilon_start, epsilon_decay=epsilon_decay, epsilon_floor=epsilon_floor,
                              policy_flip_history=policy_flip_history, episode_reward_history=episode_reward_history,
                              policy_flip_history_b=policy_flip_history_b)
    else:
        print("[Convergence] SKIPPED -- pass delta_history to enable")
    print()

    if first_episode_reward_trace is not None and last_episode_reward_trace is not None:
        plot_episode_trajectory(first_episode_reward_trace, last_episode_reward_trace)
        print()

    if q_mean_history is not None and q_max_history is not None:
        plot_q_value_trend(q_mean_history, q_max_history)
        print()

    if reward_mean_history is not None and reward_max_history is not None:
        plot_reward_trend(reward_mean_history, reward_max_history)
        print()

    if q_gap_mean_history is not None and q_gap_max_history is not None:
        plot_q_gap_trend(q_gap_mean_history, q_gap_max_history)
        print()

    if seed_agents is not None:
        check_policy_stability(seed_agents)
        print()
        check_state_action_summary(seed_agents)
    else:
        print("[Policy stability] SKIPPED -- pass seed_agents (list) to enable")
    print()

    print("=" * 60)
