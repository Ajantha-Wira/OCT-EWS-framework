 # =============================================================
# OCT Early Warning System
# main.py
#
# Purpose: Entry point for the EWS system.
# Handles phase selection and loads the correct configuration.
# Run this file to start the system.
# =============================================================

from config import (
    get_default_phase1_config,
    get_phase2_config_template,
    PhaseConfig,
)


def select_phase() -> PhaseConfig:
    """
    Asks the user to select Phase 1 or Phase 2 at runtime.
    Returns the appropriate PhaseConfig object.
    """
    print("\n" + "=" * 60)
    print("  OCT Early Warning System")
    print("=" * 60)
    print("\n  Select operating mode:\n")
    print("  1 = Phase 1: Preliminary research mode")
    print("      (provisional thresholds, research output only)")
    print()
    print("  2 = Phase 2: Validated operational mode")
    print("      (fixed clinical thresholds, patient monitoring)")
    print()

    while True:
        choice = input("  Enter 1 or 2: ").strip()

        if choice == "1":
            config = get_default_phase1_config()
            print("\n  Phase 1 selected: Preliminary Research Mode")
            print(f"\n  DISCLAIMER: {config.disclaimer}")
            print()
            return config

        elif choice == "2":
            config = get_phase2_config_template()
            print("\n  Phase 2 selected: Validated Operational Mode")
            print("\n  NOTE: Ensure validated thresholds are loaded")
            print("        before running patient assessments.")
            print()
            return config

        else:
            print("  Invalid choice. Please enter 1 or 2.")


def print_config_summary(config: PhaseConfig) -> None:
    """
    Prints a summary of the active configuration.
    """
    print("=" * 60)
    print("  Active Configuration Summary")
    print("=" * 60)
    print(f"  Phase       : {config.phase}")
    print(f"  Method      : {config.atypicality.method}")
    print(f"  Core normal : below {config.atypicality.core_normal_pct}th percentile")
    print(f"  Extended    : below {config.atypicality.extended_normal_pct}th percentile")
    print(f"  Atypical    : below {config.atypicality.atypical_candidate_pct}th percentile")
    print(f"  Suspicious  : above {config.atypicality.atypical_candidate_pct}th percentile")
    print()
    print("  Direction thresholds:")
    for cls, dcfg in config.direction_thresholds.items():
        print(f"    {cls:8s} alignment={dcfg.alignment_threshold:.2f}  "
              f"projection={dcfg.projection_threshold:.2f}  "
              f"urgency={dcfg.urgency}")
    print()
    print(f"  Budget:")
    print(f"    Immediate review : {config.budget.immediate_review_rate*100:.0f}%")
    print(f"    Deferred review  : {config.budget.deferred_review_rate*100:.0f}%")
    print(f"    Review cycle     : {config.budget.review_cycle_days} days")
    print(f"    Overflow action  : {config.budget.overflow_action}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    config = select_phase()
    print_config_summary(config)
    print("  System ready. Modules will be loaded as needed.")
    print()
