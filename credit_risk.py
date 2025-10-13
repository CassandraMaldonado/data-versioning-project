import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import uuid
import warnings
warnings.filterwarnings('ignore')

# data lineage tracking.
try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.run import RunEvent, RunState, Run, Job, Dataset
    from openlineage.client.facet import (
        SqlJobFacet, 
        SourceCodeLocationJobFacet,
        DataQualityMetricsInputDatasetFacet,
        SchemaDatasetFacet,
        SchemaField,
        DocumentationJobFacet,
        OutputStatisticsOutputDatasetFacet
    )
    LINEAGE_AVAILABLE = True
except ImportError:
    print("OpenLineage not installed.")
    LINEAGE_AVAILABLE = False


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("-"*80)
print("Credit risk analysis with data luneage.")
print("-"*80)

# Openlineage with error handling.
LINEAGE_CONNECTED = False
if LINEAGE_AVAILABLE:
    try:
        client = OpenLineageClient(url="http://localhost:5000")
        namespace = "credit_risk_analysis"
        job_name = "credit_risk_eda_pipeline"
        run_id = str(uuid.uuid4())
        
        print("OpenLineage client initialized.")
        print(f"   Namespace: {namespace}")
        print(f"   Job: {job_name}")
        print(f"   Run ID: {run_id}")
        LINEAGE_CONNECTED = True
    except Exception as e:
        print(f"Could not connect to Marquez: {e}")
        print("   Continuing without lineage tracking.")

print("-"*80)
print("Data loading.")
print("-"*80)

column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings_account', 'employment_duration', 'installment_rate', 'personal_status',
    'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
    'housing', 'existing_credits', 'job', 'num_dependents', 'telephone', 
    'foreign_worker', 'credit_risk'
]


if LINEAGE_AVAILABLE and LINEAGE_CONNECTED:
    try:
        start_event = RunEvent(
            eventType=RunState.START,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=run_id),
            job=Job(
                namespace=namespace,
                name=job_name,
                facets={
                    "documentation": DocumentationJobFacet(
                        description="Credit Risk EDA Pipeline"
                    ),
                    "sourceCodeLocation": SourceCodeLocationJobFacet(
                        type="git",
                        url="file://credit_risk.py"
                    )
                }
            ),
            producer="credit_risk_pipeline/1.0",
            inputs=[],
            outputs=[]
        )
        client.emit(start_event)
        print("Lineage START event emitted.")
    except Exception as e:
        print(f"Failed to emit START event: {e}")

print("Loading german credit risk dataset.")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

try:
    df_raw = pd.read_csv(url, sep=' ', names=column_names, header=None)
    print(f"Dataset loaded: {df_raw.shape}")
    df_raw.to_csv('credit_risk_raw.csv', index=False)
    print("Saved to credit_risk_raw.csv")
except Exception as e:
    print(f"Could not download from UCI. Using sample data.")
    np.random.seed(42)
    n_samples = 1000
    df_raw = pd.DataFrame({
        'checking_account': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
        'duration': np.random.randint(4, 72, n_samples),
        'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
        'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49', 'A410'], n_samples),
        'credit_amount': np.random.randint(250, 18000, n_samples),
        'savings_account': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
        'employment_duration': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
        'installment_rate': np.random.randint(1, 5, n_samples),
        'personal_status': np.random.choice(['A91', 'A92', 'A93', 'A94', 'A95'], n_samples),
        'other_debtors': np.random.choice(['A101', 'A102', 'A103'], n_samples),
        'residence_since': np.random.randint(1, 5, n_samples),
        'property': np.random.choice(['A121', 'A122', 'A123', 'A124'], n_samples),
        'age': np.random.randint(19, 75, n_samples),
        'other_installments': np.random.choice(['A141', 'A142', 'A143'], n_samples),
        'housing': np.random.choice(['A151', 'A152', 'A153'], n_samples),
        'existing_credits': np.random.randint(1, 5, n_samples),
        'job': np.random.choice(['A171', 'A172', 'A173', 'A174'], n_samples),
        'num_dependents': np.random.randint(1, 3, n_samples),
        'telephone': np.random.choice(['A191', 'A192'], n_samples),
        'foreign_worker': np.random.choice(['A201', 'A202'], n_samples),
        'credit_risk': np.random.choice([1, 2], n_samples)
    })
    df_raw.to_csv('credit_risk_raw.csv', index=False)

print(f"Raw data info:")
print(f"   Rows: {len(df_raw):,}")
print(f"   Columns: {len(df_raw.columns)}")
print(f"   Memory: {df_raw.memory_usage(deep=True).sum() / 1024:.2f} kb")

print("-"*80)
print("Data transformation and cleaning.")
print("-"*80)

df_clean = df_raw.copy()

print("Decoding the categorical variables.")

checking_map = {
    'A11': 'No checking account', 'A12': '< 0 DM',
    'A13': '0 <= ... < 200 DM', 'A14': '>= 200 DM'
}
savings_map = {
    'A61': '< 100 DM', 'A62': '100 <= ... < 500 DM',
    'A63': '500 <= ... < 1000 DM', 'A64': '>= 1000 DM',
    'A65': 'Unknown/No savings'
}
credit_history_map = {
    'A30': 'No credits/All paid', 'A31': 'All paid back',
    'A32': 'Existing paid', 'A33': 'Delay in past',
    'A34': 'Critical account'
}
purpose_map = {
    'A40': 'Car (new)', 'A41': 'Car (used)',
    'A42': 'Furniture/Equipment', 'A43': 'Radio/TV',
    'A44': 'Domestic appliances', 'A45': 'Repairs',
    'A46': 'Education', 'A47': 'Vacation',
    'A48': 'Retraining', 'A49': 'Business', 'A410': 'Others'
}
employment_map = {
    'A71': 'Unemployed', 'A72': '< 1 year',
    'A73': '1 <= ... < 4 years', 'A74': '4 <= ... < 7 years',
    'A75': '>= 7 years'
}

df_clean['checking_account_decoded'] = df_clean['checking_account'].map(checking_map)
df_clean['savings_account_decoded'] = df_clean['savings_account'].map(savings_map)
df_clean['credit_history_decoded'] = df_clean['credit_history'].map(credit_history_map)
df_clean['purpose_decoded'] = df_clean['purpose'].map(purpose_map)
df_clean['employment_decoded'] = df_clean['employment_duration'].map(employment_map)
df_clean['risk_label'] = df_clean['credit_risk'].map({1: 'Good', 2: 'Bad'})

# derived features.
df_clean['age_group'] = pd.cut(df_clean['age'], 
                                bins=[0, 25, 35, 50, 100], 
                                labels=['18-25', '26-35', '36-50', '50+'])
df_clean['amount_category'] = pd.cut(df_clean['credit_amount'],
                                      bins=[0, 2000, 5000, 10000, 20000],
                                      labels=['Low (<2K)', 'Medium (2K-5K)', 
                                             'High (5K-10K)', 'Very High (>10K)'])
df_clean['duration_category'] = pd.cut(df_clean['duration'],
                                        bins=[0, 12, 24, 36, 100],
                                        labels=['Short (<1yr)', 'Medium (1-2yr)',
                                               'Long (2-3yr)', 'Very Long (>3yr)'])


print("New features created: age_group, amount_category and duration_category.")

df_clean.to_csv('credit_risk_cleaned.csv', index=False)
print("Saved to: credit_risk_cleaned.csv")


print("-"*80)
print("EDA.")
print("-"*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# credit risk distribution.
ax1 = fig.add_subplot(gs[0, 0])
risk_counts = df_clean['risk_label'].value_counts()
colors_risk = ['#2ecc71', '#e74c3c']
ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', 
        colors=colors_risk, startangle=90)
ax1.set_title('Credit risk distribution', fontweight='bold', fontsize=12)

# age distribution by risk.
ax2 = fig.add_subplot(gs[0, 1])
df_clean.boxplot(column='age', by='risk_label', ax=ax2)
ax2.set_title('Age distribution by credit risk', fontweight='bold', fontsize=12)
ax2.set_xlabel('Credit risk')
ax2.set_ylabel('Age')
plt.sca(ax2)
plt.xticks(rotation=0)

# credit amount distribution.
ax3 = fig.add_subplot(gs[0, 2])
df_clean.boxplot(column='credit_amount', by='risk_label', ax=ax3)
ax3.set_title('Credit amount by risk', fontweight='bold', fontsize=12)
ax3.set_xlabel('Credit risk')
ax3.set_ylabel('Credit amount (DM)')
plt.sca(ax3)
plt.xticks(rotation=0)

# duration distribution.
ax4 = fig.add_subplot(gs[0, 3])
df_clean.boxplot(column='duration', by='risk_label', ax=ax4)
ax4.set_title('Loan duration by risk', fontweight='bold', fontsize=12)
ax4.set_xlabel('Credit risk')
ax4.set_ylabel('Duration (months)')
plt.sca(ax4)
plt.xticks(rotation=0)

# purpose distribution.
ax5 = fig.add_subplot(gs[1, :2])
purpose_risk = pd.crosstab(df_clean['purpose_decoded'], df_clean['risk_label'])
purpose_risk.plot(kind='bar', stacked=False, ax=ax5, color=['#2ecc71', '#e74c3c'])
ax5.set_title('Loan purpose by credit risk', fontweight='bold', fontsize=12)
ax5.set_xlabel('Purpose')
ax5.set_ylabel('Count')
ax5.legend(title='Risk')
plt.sca(ax5)
plt.xticks(rotation=45, ha='right')

# checking account status.
ax6 = fig.add_subplot(gs[1, 2:])
checking_risk = pd.crosstab(df_clean['checking_account_decoded'], df_clean['risk_label'])
checking_risk.plot(kind='bar', stacked=False, ax=ax6, color=['#2ecc71', '#e74c3c'])
ax6.set_title('Checking account status by risk', fontweight='bold', fontsize=12)
ax6.set_xlabel('Checking account status')
ax6.set_ylabel('Count')
ax6.legend(title='Risk')
plt.sca(ax6)
plt.xticks(rotation=45, ha='right')

# age groups.
ax7 = fig.add_subplot(gs[2, 0])
age_risk = pd.crosstab(df_clean['age_group'], df_clean['risk_label'])
age_risk.plot(kind='bar', ax=ax7, color=['#2ecc71', '#e74c3c'])
ax7.set_title('Age groups by risk', fontweight='bold', fontsize=12)
ax7.set_xlabel('Age group')
ax7.set_ylabel('Count')
ax7.legend(title='Risk')
plt.sca(ax7)
plt.xticks(rotation=0)

# credit amount categories.
ax8 = fig.add_subplot(gs[2, 1])
amount_risk = pd.crosstab(df_clean['amount_category'], df_clean['risk_label'])
amount_risk.plot(kind='bar', ax=ax8, color=['#2ecc71', '#e74c3c'])
ax8.set_title('Amount categories by risk', fontweight='bold', fontsize=12)
ax8.set_xlabel('Amount category')
ax8.set_ylabel('Count')
ax8.legend(title='Risk')
plt.sca(ax8)
plt.xticks(rotation=45, ha='right')

# employment duration.
ax9 = fig.add_subplot(gs[2, 2])
employment_risk = pd.crosstab(df_clean['employment_decoded'], df_clean['risk_label'])
employment_risk.plot(kind='bar', ax=ax9, color=['#2ecc71', '#e74c3c'])
ax9.set_title('Employment by risk', fontweight='bold', fontsize=12)
ax9.set_xlabel('Employment duration')
ax9.set_ylabel('Count')
ax9.legend(title='Risk')
plt.sca(ax9)
plt.xticks(rotation=45, ha='right')

# correlation heatmap.
ax10 = fig.add_subplot(gs[2, 3])
numeric_cols = ['age', 'credit_amount', 'duration', 'installment_rate', 'credit_risk']
corr_matrix = df_clean[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax10, 
            cbar_kws={'shrink': 0.8})
ax10.set_title('Correlation matrix', fontweight='bold', fontsize=12)

plt.savefig('credit_risk_eda_report.png', dpi=300, bbox_inches='tight')
plt.close()


print("-"*80)
print("Insights and stats.")
print("-"*80)

insights = []

# overall risk distribution.
good_pct = (df_clean['risk_label'] == 'Good').sum() / len(df_clean) * 100
bad_pct = (df_clean['risk_label'] == 'Bad').sum() / len(df_clean) * 100
insights.append(f"Credit portfolio: {good_pct:.1f}% Good credit, {bad_pct:.1f}% Bad credit.")

# average credit amount by risk.
avg_good = df_clean[df_clean['risk_label'] == 'Good']['credit_amount'].mean()
avg_bad = df_clean[df_clean['risk_label'] == 'Bad']['credit_amount'].mean()
insights.append(f"Avg credit amount: Good={avg_good:,.0f} DM, Bad={avg_bad:,.0f} DM.")

# age analysis.
avg_age_good = df_clean[df_clean['risk_label'] == 'Good']['age'].mean()
avg_age_bad = df_clean[df_clean['risk_label'] == 'Bad']['age'].mean()
insights.append(f"Avg age: Good={avg_age_good:.1f} years, Bad={avg_age_bad:.1f} years.")

# duration analysis.
avg_dur_good = df_clean[df_clean['risk_label'] == 'Good']['duration'].mean()
avg_dur_bad = df_clean[df_clean['risk_label'] == 'Bad']['duration'].mean()
insights.append(f"Avg duration: Good={avg_dur_good:.1f} months, Bad={avg_dur_bad:.1f} months.")

# purpose with highest risk.
purpose_risk_rate = df_clean.groupby('purpose_decoded')['credit_risk'].apply(
    lambda x: (x == 2).sum() / len(x) * 100
).sort_values(ascending=False)
insights.append(f"Highest risk purpose: {purpose_risk_rate.index[0]} ({purpose_risk_rate.iloc[0]:.1f}% bad).")

# checking account impact.
checking_risk_rate = df_clean.groupby('checking_account_decoded')['credit_risk'].apply(
    lambda x: (x == 2).sum() / len(x) * 100
).sort_values(ascending=False)
insights.append(f"Highest risk checking status: {checking_risk_rate.index[0]} ({checking_risk_rate.iloc[0]:.1f}% bad).")

# age group analysis.
age_group_risk = df_clean.groupby('age_group')['credit_risk'].apply(
    lambda x: (x == 2).sum() / len(x) * 100
).sort_values(ascending=False)
insights.append(f"Highest risk age group: {age_group_risk.index[0]} ({age_group_risk.iloc[0]:.1f}% bad).")

# total credit exposure.
total_credit = df_clean['credit_amount'].sum()
insights.append(f"Total credit exposure: {total_credit:,.0f} DM.")

# avg installment rate.
avg_installment = df_clean['installment_rate'].mean()
insights.append(f"Avg installment rate: {avg_installment:.1f}% of disposable income.")

# employment impact.
employment_risk_rate = df_clean.groupby('employment_decoded')['credit_risk'].apply(
    lambda x: (x == 2).sum() / len(x) * 100
).sort_values(ascending=False)
insights.append(f"Highest risk employment: {employment_risk_rate.index[0]} ({employment_risk_rate.iloc[0]:.1f}% bad).")

print("Insights:")
for i, insight in enumerate(insights, 1):
    print(f"  {i}. {insight}")

with open('credit_risk_insights.txt', 'w') as f:
    f.write("Credit risk analysis insights.")
    f.write("-"*80 + "\n\n")
    for i, insight in enumerate(insights, 1):
        f.write(f"{i}. {insight}\n")

print("Saved to: credit_risk_insights.txt")

# summary statistics table.
summary_stats = df_clean.groupby('risk_label').agg({
    'age': ['mean', 'median', 'std'],
    'credit_amount': ['mean', 'median', 'std'],
    'duration': ['mean', 'median', 'std']
}).round(2)

print("Summary stats by risk:")
print(summary_stats)

summary_stats.to_csv('credit_risk_summary_stats.csv')
print("Summary stats saved to: credit_risk_summary_stats.csv")


print("-"*80)
print("Data lineage.")
print("-"*80)


fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')


ax.text(5, 9.5, 'Data lineage diagram', ha='center', fontsize=20, fontweight='bold')
ax.text(5, 9, 'Credit risk analysis pipeline', ha='center', fontsize=14, style='italic')


color_source = '#3498db'
color_transform = '#e67e22'
color_analysis = '#9b59b6'
color_output = '#27ae60'

# data source.
ax.add_patch(plt.Rectangle((0.5, 7), 2, 1, facecolor=color_source, edgecolor='black', linewidth=2))
ax.text(1.5, 7.5, 'DATA SOURCE\n\nGerman Credit Data\n(UCI Repository)\n1000 records, 21 attributes', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')


ax.annotate('', xy=(3.5, 7.5), xytext=(2.5, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(3, 7.8, 'Extract', ha='center', fontsize=9, style='italic')

# raw data storage.
ax.add_patch(plt.Rectangle((3.5, 7), 2, 1, facecolor=color_source, edgecolor='black', linewidth=2))
ax.text(4.5, 7.5, 'Raw data\n\ncredit_risk_raw.csv\nOriginal format\nCoded categories', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')


ax.annotate('', xy=(4.5, 6.8), xytext=(4.5, 6),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(5, 6.4, 'Transform', ha='center', fontsize=9, style='italic')

# transformation layer.
transform_box_y = 4.5
ax.add_patch(plt.Rectangle((1, transform_box_y), 7, 1.3, facecolor=color_transform, 
                          edgecolor='black', linewidth=2))
ax.text(4.5, transform_box_y + 1, 'TRANSFORMATION LAYER', ha='center', fontsize=12, 
        fontweight='bold', color='white')


transforms = [
    '1. Decode categories',
    '2. Create age_group',
    '3. Create amount_category',
    '4. Create duration_category',
    '5. Map risk labels'
]
transform_text = '\n'.join(transforms)
ax.text(4.5, transform_box_y + 0.35, transform_text, ha='center', va='center', 
        fontsize=8, color='white')


ax.annotate('', xy=(4.5, 4.3), xytext=(4.5, 3.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(5, 3.9, 'Load', ha='center', fontsize=9, style='italic')

# cleaned data.
ax.add_patch(plt.Rectangle((3.5, 2.5), 2, 0.8, facecolor=color_source, edgecolor='black', linewidth=2))
ax.text(4.5, 2.9, 'CLEANED DATA\n\ncredit_risk_cleaned.csv', 
        ha='center', va='center', fontsize=10, fontweight='bold', color='white')


ax.annotate('', xy=(1.5, 2.9), xytext=(3.5, 2.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(2.5, 3.2, 'Analyze', ha='center', fontsize=9, style='italic')


ax.annotate('', xy=(7, 2.9), xytext=(5.5, 2.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(6.2, 3.2, 'Analyze', ha='center', fontsize=9, style='italic')


# EDA.
ax.add_patch(plt.Rectangle((0.2, 1), 2.5, 1.2, facecolor=color_analysis, edgecolor='black', linewidth=2))
ax.text(1.45, 1.9, 'EXPLORATORY DATA\nANALYSIS', ha='center', fontsize=10, 
        fontweight='bold', color='white')
ax.text(1.45, 1.4, '• Distribution analysis\n• Risk segmentation\n• Statistical summaries\n• Correlations', 
        ha='center', va='center', fontsize=8, color='white')

# insights.
ax.add_patch(plt.Rectangle((6.8, 1), 2.5, 1.2, facecolor=color_analysis, edgecolor='black', linewidth=2))
ax.text(8.05, 1.9, 'BUSINESS INSIGHTS', ha='center', fontsize=10, 
        fontweight='bold', color='white')
ax.text(8.05, 1.4, '• Risk patterns\n• Key metrics\n• Recommendations\n• Summary stats', 
        ha='center', va='center', fontsize=8, color='white')


ax.annotate('', xy=(1.45, 0.8), xytext=(1.45, 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.annotate('', xy=(8.05, 0.8), xytext=(8.05, 1),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# outputs.
ax.add_patch(plt.Rectangle((0.2, 0.1), 2.5, 0.6, facecolor=color_output, edgecolor='black', linewidth=2))
ax.text(1.45, 0.4, 'VISUALIZATIONS\n\ncredit_risk_eda_report.png', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# reports.
ax.add_patch(plt.Rectangle((6.8, 0.1), 2.5, 0.6, facecolor=color_output, edgecolor='black', linewidth=2))
ax.text(8.05, 0.4, 'REPORTS & INSIGHTS\n\ninsights.txt, summary_stats.csv', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='white')


legend_y = 8.5
ax.add_patch(plt.Rectangle((7.5, legend_y - 0.2), 0.3, 0.2, facecolor=color_source, edgecolor='black'))
ax.text(8, legend_y - 0.1, 'Data Storage', ha='left', va='center', fontsize=9)

ax.add_patch(plt.Rectangle((7.5, legend_y - 0.5), 0.3, 0.2, facecolor=color_transform, edgecolor='black'))
ax.text(8, legend_y - 0.4, 'Transformation', ha='left', va='center', fontsize=9)

ax.add_patch(plt.Rectangle((7.5, legend_y - 0.8), 0.3, 0.2, facecolor=color_analysis, edgecolor='black'))
ax.text(8, legend_y - 0.7, 'Analysis', ha='left', va='center', fontsize=9)

ax.add_patch(plt.Rectangle((7.5, legend_y - 1.1), 0.3, 0.2, facecolor=color_output, edgecolor='black'))
ax.text(8, legend_y - 1, 'Output', ha='left', va='center', fontsize=9)


metadata_text = f"""Pipeline Metadata:
• Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
• Total Records: {len(df_clean):,}
• Source: UCI ML Repository
• Tool: Python + Pandas
• Lineage: OpenLineage"""

ax.text(0.2, 7.2, metadata_text, ha='left', va='top', fontsize=8, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('credit_risk_data_lineage.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Data lineage diagram saved: credit_risk_data_lineage.png")


if LINEAGE_AVAILABLE and LINEAGE_CONNECTED:
    print("\n" + "="*80)
    print("Emitting data lineage to Marquez.")
    print("="*80)
    
    try:
        # input dataset.
        input_dataset = Dataset(
            namespace=namespace,
            name="uci_german_credit_raw",
            facets={
                "schema": SchemaDatasetFacet(
                    fields=[
                        SchemaField(name=col, type="string" if df_raw[col].dtype == 'object' else "integer")
                        for col in df_raw.columns[:10]
                    ]
                ),
                "dataQuality": DataQualityMetricsInputDatasetFacet(
                    rowCount=len(df_raw),
                    bytes=int(df_raw.memory_usage(deep=True).sum()),
                    columnMetrics={}
                )
            }
        )
        
        # output datasets.
        output_cleaned = Dataset(
            namespace=namespace,
            name="credit_risk_cleaned",
            facets={
                "schema": SchemaDatasetFacet(
                    fields=[
                        SchemaField(name="age_group", type="category"),
                        SchemaField(name="amount_category", type="category"),
                        SchemaField(name="duration_category", type="category"),
                        SchemaField(name="risk_label", type="string")
                    ]
                ),
                "outputStatistics": OutputStatisticsOutputDatasetFacet(
                    rowCount=len(df_clean),
                    size=int(df_clean.memory_usage(deep=True).sum())
                )
            }
        )
        
        output_insights = Dataset(
            namespace=namespace,
            name="credit_risk_insights",
            facets={
                "documentation": DocumentationJobFacet(
                    description="Business insights and key metrics from credit risk analysis"
                )
            }
        )
        
        output_visualizations = Dataset(
            namespace=namespace,
            name="credit_risk_eda_report",
            facets={
                "documentation": DocumentationJobFacet(
                    description="Comprehensive EDA visualizations"
                )
            }
        )
        

        complete_event = RunEvent(
            eventType=RunState.COMPLETE,
            eventTime=datetime.now().isoformat(),
            run=Run(runId=run_id),
            job=Job(
                namespace=namespace,
                name=job_name,
                facets={
                    "documentation": DocumentationJobFacet(
                        description="Credit Risk EDA Pipeline completed"
                    ),
                    "sql": SqlJobFacet(
                        query="""
-- Transformations Applied:
-- 1. Decoded categorical variables (checking_account, savings, etc.)
-- 2. Created age_group feature (bins: 18-25, 26-35, 36-50, 50+)
-- 3. Created amount_category (Low, Medium, High, Very High)
-- 4. Created duration_category (Short, Medium, Long, Very Long)
-- 5. Mapped risk labels (1=Good, 2=Bad)
                        """
                    )
                }
            ),
            producer="credit_risk_pipeline/1.0",
            inputs=[input_dataset],
            outputs=[output_cleaned, output_insights, output_visualizations]
        )
        
        client.emit(complete_event)
        print(f"Lineage Summary:")
        print(f"   Input: {input_dataset.name}")
        print(f"   Outputs: {len([output_cleaned, output_insights, output_visualizations])} datasets")
        print(f"   Transformations: 5 major steps")
        print(f"   View lineage at: http://localhost:3000")
        print(f"   Job: {namespace}/{job_name}")
        print(f"   Run ID: {run_id}")
        
    except Exception as e:
        print(f"Failed to emit complete event: {e}")
        print(" Lineage diagram was still created")


print("-"*80)
print("Pipeline complete.")
print("-"*80)

print("Output Files Generated:")
print("   - credit_risk_raw.csv is the raw data from UCI.")
print("   - credit_risk_cleaned.csv is the cleaned and transformed data.")
print("   - credit_risk_insights.txt id the key business insights.")
print("   - credit_risk_summary_stats.csv is the stats summaries.")
print("   - credit_risk_eda_report.png is the EDA visuals.")
print("   - credit_risk_data_lineage.png is the data lineage diagram.")

if LINEAGE_AVAILABLE and LINEAGE_CONNECTED:
    print("Data lineage tracking:")
    print("  - OpenLineage events emitted to Marquez")
    print("  - Access UI: http://localhost:3000")
    print(f" - Namespace: {namespace}")
    print(f" - Job: {job_name}")
    print(f" - Run ID: {run_id}")
else:
    print("Data lineage:")
    print(" - Visual diagram created (credit_risk_data_lineage.png)")
    print(" - To track with Marquez, run:")
    print("      docker run -d -p 3000:3000 -p 5000:5000 marquezproject/marquez")

print("Key stats:")
print(f"   - Total records: {len(df_clean):,}")
print(f"   - Good credit: {good_pct:.1f}%")
print(f"   - Bad credit: {bad_pct:.1f}%")
print(f"   - Total exposure: {total_credit:,.0f} DM")


if LINEAGE_AVAILABLE and LINEAGE_CONNECTED:
    print("Explore lineage in Marquez UI: http://localhost:3000")
