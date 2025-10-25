#%% 0: Imports ################
import pandas as pd
import numpy as np

###########################################################################
#%% 1: Define functions
###########################################################################

def prepareIOTable(pivot_io_table, target_countries, industry_codes, final_demand_codes, value_added_codes, dom_sector):
    ''' 
    Prepare relevant IO Table components for given target countries from pivoted FIGARO table.
    '''
    # Select intermediate demand for target_countries (MultiIndex preserved)
    # Extract Intermediate Demand (Z)
    Z_ij = pivot_io_table.loc[(target_countries, industry_codes), (target_countries, industry_codes)]
    Z_ij.name = 'Intermediate Demand (Z_ij), in mEUR'

    # Final Demand (x)
    x_columns = pivot_io_table.loc[target_countries, (target_countries, final_demand_codes)]

    # Imports
    imports_mask = ~pivot_io_table.index.get_level_values(0).isin(target_countries + [dom_sector])
    imports = pivot_io_table.loc[imports_mask, target_countries]
    Im_r = imports.sum()

    # Exports
    exports_mask = ~pivot_io_table.columns.get_level_values(0).isin(target_countries)
    exports = pivot_io_table.loc[target_countries, exports_mask]
    Ex_i = exports.sum(axis=1)

    # Sum across all final demand uses per sector (i.e., row)
    x_i = x_columns.sum(axis=1) + Ex_i.reindex(x_columns.index).fillna(0)
    x_i.name = 'Final Demand, in mEUR'

    # Value Added (v)
    v_rows = pivot_io_table.loc[(dom_sector, value_added_codes), (target_countries, industry_codes)]
    v_j = v_rows.sum()
    v_j.index = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in v_j.index], names=['counterpartArea', 'colIi'])
    v_j += Im_r.reindex(v_j.index).fillna(0)
    v_j.name = 'Value Added, in mEUR'

    # Total Output (q)
    q_i = Z_ij.sum(axis=1) + x_i
    q_i.name = 'Total Output Revenues, in mEUR'
    q_j = Z_ij.sum(axis=0) + v_j
    q_j.name = 'Total Outlay Expenses, in mEUR'

    if not np.isclose(q_j.sum(), q_i.sum()):
        raise ValueError("Total output revenues (q_i) and outlay expenses (q_j) do not match.")

    # Technical Coefficients matrix (A)
    A_ij = Z_ij.divide(q_j, axis=1)
    A_ij = A_ij.fillna(0)
    A_ij.name = 'Technical Coefficients (A_ij), in mEUR'

    q_test = A_ij.dot(q_j) + x_i
    if not np.isclose(q_test, q_i).all():
        raise ValueError("Total output revenues (q_i) do not match after constructing A_ij.")

    # Leontief Inverse (L)
    I = np.eye(len(A_ij))
    L_ij = pd.DataFrame(
        np.linalg.inv(I - A_ij.values),
        index=A_ij.index,
        columns=A_ij.columns
    )

    # Primary input coefficients (l')
    l_j = v_j / q_j
    l_j.name = 'Primary Input Coefficients (l_i), in mEUR'
    l_j = l_j.fillna(1)

    p = l_j.dot(L_ij)
    if not np.allclose(p, 1):
        raise ValueError("Not all values in p are close to 1.")

    # Assemble IO Table
    IO_table = {
        'Z_ij': Z_ij,
        'A_ij': A_ij,
        'L_ij': L_ij,
        'x_i': x_i,
        'v_j': v_j,
        'q_i': q_i,
        'q_j': q_j,
        'l_j': l_j,
        'Im_r': Im_r,
        'Ex_i': Ex_i
    }
    return IO_table

###########################################################################
#%% 2: Prepare MRIO Table from Figaro Data
###########################################################################
### 1.1: Load Data and prepare IO Table ###
# 2019 data
df = pd.read_csv('Figaro Data\estat_naio_10_fcp_ii3.tsv', sep='\t')
df_2019 = df[['freq,ind_use,ind_ava,c_dest,unit,c_orig\TIME_PERIOD', '2019 ']]

# Split the composite index column into separate columns
df_2019 = df_2019['freq,ind_use,ind_ava,c_dest,unit,c_orig\TIME_PERIOD'].str.split(",", expand=True)
# Add the 2019 data column back to the dataframe
df_2019 = pd.concat([df_2019, df['2019 ']], axis=1)
# Rename columns
df_2019.columns = ['freq', 'colIi', 'rowIi', 'counterpartArea', 'unit', 'refArea', 'obsValue']
df_2019 = df_2019.drop(columns=['freq', 'unit'])

df_2019_small = df_2019[['refArea', 'rowIi', 'counterpartArea', 'colIi', 'obsValue']]

# rename industry codes to match 2022 data
cols_to_fix = ["rowIi", "colIi"]
df_2019_small[cols_to_fix] = df_2019_small[cols_to_fix].apply(lambda col: col.str.replace("-", "T", regex=False))

# Create pivot table
pivot_io_table_2019 = df_2019_small.pivot_table(
    index=['refArea', 'rowIi'], 
    columns=['counterpartArea', 'colIi'], 
    values='obsValue',
    aggfunc='sum',
    fill_value=np.nan
)

# Save pivoted IO table
pivot_io_table_2019.to_csv('Prepared Data/pivot_io_table_2019.csv')

###########################################################################
#%% 2: Prepare IO Tables
###########################################################################

### 2.1: Load Pivot IO Table ###
pivot_io_table = pd.read_csv('Prepared Data/pivot_io_table_2019.csv', index_col=[0,1], header=[0,1])


### 2.2 Define codes ###
# Final demand codes
final_demand_codes = ['P3_S13', 'P3_S14', 'P3_S15', 'P51G', 'P5M']
# Value added codes
value_added_codes = ['D21X31', 'OP_NRES', 'OP_RES', 'D1', 'D29X39', 'B2A3G']
# industry codes
industry_codes = list(set(pivot_io_table.columns.get_level_values(1).unique()) - set(final_demand_codes))
industry_codes.sort()

### 2.3: Prepare IO Table for Germany ###
io_table_DE = prepareIOTable(pivot_io_table=pivot_io_table, target_countries = ['DE'], industry_codes=industry_codes, final_demand_codes=final_demand_codes, value_added_codes=value_added_codes, dom_sector='DOM')

### 2.4: Prepare IO Table for EU ###
# Get list of EU country codes from Figaro description file
EU_codes_list = pd.read_excel('Figaro Data/Description_FIGARO_Tables(24ed).xlsx', sheet_name='Geographical areas', usecols='B', skiprows=5).squeeze().tolist()
EU_codes_list.sort()

io_table_EU = prepareIOTable(pivot_io_table=pivot_io_table, target_countries = EU_codes_list, industry_codes=industry_codes, final_demand_codes=final_demand_codes, value_added_codes=value_added_codes, dom_sector='DOM')

###########################################################################
#%% 3: Prepare AirEmission Accounts for Germany
###########################################################################

### 3.1: Prepare Air Emission Data for EU 2022 ###
# Load Air Emission and Codes Data ###
df_emission_EU_2019_original = pd.read_excel('Figaro Data\Air Emission Accounts\env_ac_ainah_r2_EU_2019.xlsx', sheet_name='Sheet 1', skiprows=8, header=[0])
country_code = pd.read_excel('Figaro Data/Description_FIGARO_Tables(24ed).xlsx', sheet_name='Geographical areas', usecols='B:C', skiprows=5)
nace_code = pd.read_excel('Figaro Data/Description_FIGARO_Tables(24ed).xlsx', sheet_name='Prod, Ind & Accounting items', usecols='E:F', skiprows=5)

df_emission_EU_2019 = df_emission_EU_2019_original.copy()
df_emission_EU_2019.columns = ['refArea', 'rowIi'] + [str(col) + ', in Tonnes' for col in df_emission_EU_2019.columns[2:]]

# Drop empty rows
df_emission_EU_2019 = df_emission_EU_2019.drop(0)
df_emission_EU_2019 = df_emission_EU_2019.drop(df_emission_EU_2019.index[-3:])

# Rename countries and industries to match IO Table
country_code_dict = dict(zip(country_code['Label'], country_code['Code']))
country_code_dict['Greece'] = 'GR'  # Rename Greece
country_code_dict.pop('Czech Republic', None)
country_code_dict['Czechia'] = 'CZ'  # Rename Czechia

df_emission_EU_2019.refArea = df_emission_EU_2019.refArea.replace(country_code_dict)

nace_code_dict = dict(zip(nace_code['Label.1'], nace_code['Code.1']))
df_emission_EU_2019.rowIi = df_emission_EU_2019.rowIi.replace(nace_code_dict)

# Drop all rows where rowIi is not in industry_codes
df_emission_EU_2019 = df_emission_EU_2019[df_emission_EU_2019.rowIi.isin(industry_codes)]

# Check for nan values
if df_emission_EU_2019.isnull().values.any():
    raise ValueError("DataFrame contains NaN values.")

# Set the first two columns ['refArea', 'rowIi'] as index and drop refArea columns
df_emission_EU_2019 = df_emission_EU_2019.reset_index(drop=True).set_index(['refArea', 'rowIi']).sort_index()

# Add Total Emissions column in CO2e
df_emission_EU_2019['Carbon dioxide equivalent, in Tonnes'] = df_emission_EU_2019[
    ['Carbon dioxide, in Tonnes', 'Methane (CO2 equivalent), in Tonnes', 'Nitrous oxide (CO2 equivalent), in Tonnes']
].sum(axis=1)


###########################################################################
#%% 4: Save results
###########################################################################

### 4.1: Save DE IO Table to HDF5 file ###
with pd.HDFStore('Prepared Data/IO_DE_2019.h5') as store:
    for key, df in io_table_DE.items():
        store.put(key, df)

### 4.2: Save EU IO Table to HDF5 file ###
with pd.HDFStore('Prepared Data/IO_EU_2019.h5') as store:
    for key, df in io_table_EU.items():
        store.put(key, df)

### 4.3: Save EU Air Emission Data to HDF5 file ###
with pd.HDFStore('Prepared Data/AirEmissionAccounts_Tonnes.h5') as store:
    store.put('AEA_EU_2019', df_emission_EU_2019)

###########################################################################