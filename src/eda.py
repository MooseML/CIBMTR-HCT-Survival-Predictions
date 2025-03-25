import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter

class EDA:
    def __init__(self, color, data):
        self._color = color
        self.data = data

    def distribution_plot(self, col, title, bins=50):
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[col], bins=bins, color=self._color, edgecolor='black', alpha=0.7)
        plt.title(title, fontsize=16, weight='bold')
        plt.xlabel('Values', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlim(self.data[col].min(), self.data[col].max())
        plt.tight_layout()
        plt.show()

    def plot_cv(self, scores, title, metric='Stratified C-Index'):
        """
        Plot cross-validation scores with mean, std, and confidence interval.
        """
        fold_scores = np.round(scores, 3)
        mean_score = np.round(np.mean(scores), 3)
        std_score = np.round(np.std(scores), 3)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(fold_scores) + 1), fold_scores, marker='o', color=self._color, label=f'{metric} per Fold')

        for i, score in enumerate(fold_scores, start=1):
            plt.text(i, score, f'{score:.3f}', fontsize=10, ha='center', va='bottom')

        plt.axhline(mean_score, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_score:.3f}')
        plt.fill_between(range(1, len(fold_scores) + 1), mean_score - std_score, mean_score + std_score,
                         color='gray', alpha=0.2, label=f'±1 STD ({std_score:.3f})')

        plt.title(f'{title} | CV {metric}', fontsize=16, weight='bold')
        plt.xlabel('Fold', fontsize=14)
        plt.ylabel(f'{metric} Score', fontsize=14)
        plt.xticks(range(1, len(fold_scores) + 1))
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_roc_auc_logloss(self, auc_scores, logloss_scores, title):
        """
        Plot ROC AUC and Log Loss scores for classification models.
        """
        folds = range(1, len(auc_scores) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Fold', fontsize=14)
        ax1.set_ylabel('ROC AUC', color=color, fontsize=14)
        ax1.plot(folds, auc_scores, marker='o', linestyle='-', color=color, label='ROC AUC')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)

        for i, score in enumerate(auc_scores, start=1):
            ax1.text(i, score, f'{score:.3f}', fontsize=10, ha='center', va='bottom', color=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Log Loss', color=color, fontsize=14)
        ax2.plot(folds, logloss_scores, marker='s', linestyle='--', color=color, label='Log Loss')
        ax2.tick_params(axis='y', labelcolor=color)

        for i, score in enumerate(logloss_scores, start=1):
            ax2.text(i, score, f'{score:.3f}', fontsize=10, ha='center', va='top', color=color)

        fig.tight_layout()
        plt.title(f'{title} | ROC AUC & Log Loss per Fold', fontsize=16, weight='bold')
        plt.show()

    def plot_kaplan_meier(self, time_col='efs_time', event_col='efs'):
        kmf = KaplanMeierFitter()
        kmf.fit(self.data[time_col], event_observed=self.data[event_col])
        plt.figure(figsize=(10, 6))
        kmf.plot_survival_function(color=self._color)
        plt.title('Kaplan-Meier Survival Curve', fontsize=16, weight='bold')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Survival Probability', fontsize=14)
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()

    def plot_nelson_aalen(self, time_col='efs_time', event_col='efs'):
        naf = NelsonAalenFitter()
        naf.fit(self.data[time_col], event_observed=self.data[event_col])
        plt.figure(figsize=(10, 6))
        naf.plot_cumulative_hazard(color=self._color)
        plt.title('Nelson-Aalen Cumulative Hazard Curve', fontsize=16, weight='bold')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Cumulative Hazard', fontsize=14)
        plt.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()
