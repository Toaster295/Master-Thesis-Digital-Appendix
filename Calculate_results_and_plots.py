#%% 0: Imports ################
import pandas as pd
import numpy as np
from collections import OrderedDict
import os
import sys
import matplotlib.pyplot as plt


###########################################################################
#%% 1: Load Data and define Parameters
###########################################################################
### 1.1: Load IO Tables ###
# Germany only
with pd.HDFStore('Prepared Data/IO_DE_2019.h5', 'r') as store:
    IO_data_DE = {
        'L_ij': store['L_ij'],
        'x_i': store['x_i'],
        'v_j': store['v_j'],
        'q_j': store['q_j'],
        'l_j': store['l_j']
    }
# EU27
with pd.HDFStore('Prepared Data/IO_EU_2019.h5', 'r') as store:
    IO_data_EU = {
        'L_ij': store['L_ij'],
        'x_i': store['x_i'],
        'v_j': store['v_j'],
        'q_j': store['q_j'],
        'l_j': store['l_j']
    }

### 1.2: Load Air Emission Data ###
with pd.HDFStore('Prepared Data/AirEmissionAccounts_Tonnes.h5', 'r') as store:
    df_emissions = store['AEA_EU_2019']
    
### 1.3: Create Sector Mapping ###
nace_name = pd.read_excel('Figaro Data/Description_FIGARO_Tables(24ed).xlsx', sheet_name='Prod, Ind & Accounting items', usecols='E:F', skiprows=5)
nace_name_dict = dict(zip(nace_name['Code.1'], nace_name['Label.1']))

country_name = pd.read_excel('Figaro Data/Description_FIGARO_Tables(24ed).xlsx', sheet_name='Geographical areas', usecols='B:C', skiprows=5)
country_name_dict = dict(zip(country_name['Code'], country_name['Label']))
country_name_dict['GR'] = 'Greece'  # Rename Greece

### 1.4: Load Elasticity Data ###
elasticity_data = pd.Series(pd.read_excel('Elasticities/own_price_elasticities-NACE-clean.xlsx', index_col=0)['Own-price elasticity (all households)'])
# Prepare Elasticity Vector
elasticity_data.index = IO_data_DE['q_j'].index # Ensure same index as q_j
elasticity_data.mean() # -0.2548780487804878
etha_j = elasticity_data.fillna(-0.25) # Fill missing sectors with mean elasticity
# EU wide elasticity vector (only germany values, rest nan)
etha_j_EU = pd.Series(np.nan, index=IO_data_EU['q_j'].index)
etha_j_EU.update(etha_j) # fill in germany values


### 1.5: Load and Prepare Carbon Tax Coverage Data ###
coverage_data = pd.read_excel('Carbon Tax Prices/CarbonTax_Coverage_Clean.xls', sheet_name='Coverage', index_col=[0,1])
coverage_data_YorP = coverage_data[coverage_data['Status_CO2'].isin(['Y', 'Partial'])]
coverage_data_YorP = coverage_data_YorP.drop(columns=['Status_CO2'])
coverage_data_prepared = pd.DataFrame(0.0, index=IO_data_EU['q_j'].index, columns=coverage_data_YorP.columns)
coverage_data_prepared.update(coverage_data_YorP)
# coverage source for comparison
coverage_source = pd.read_excel('Carbon Tax Prices\EU CO2 Tax prices 2019.xls')

### 1.6: Define Parameters ###
# tax rates
tax_rates = [65, 120, 171] # Conversion 200$ with 1.1672$/€ at 26.9.2025

# Interest rate for amortization calculations
interest_rate = 0.02 # 2% interest rate

# Emission reduction assumed for amortization and emission savings calculations
emission_reduction = 0.1 # 10% emission reduction through efficiency improvements

# Emission type for calculations
emission_column = 'CO2e Emissions (mt)'  # Options: 'CO2e Emissions (mt)' or 'CO2 Emissions (mt)'

# Amortization times to consider
amortization_times_range = np.arange(1, 21) # 1 to 20 years

###########################################################################
#%% 2: define Functions
###########################################################################
##### Define Calculation Functions for Tax Overviews and Effects #####
def CalculateTaxOverview(
    tax_rate, 
    countries=['DE'], 
    sectors=slice(None), # all sectors
    emission_column=emission_column, 
    IO_data=IO_data_DE, 
    df_emissions=df_emissions):
    
    """
    Calculate tax overview for a specific countries and sectors (Used mainly for Germany - all sectors).
    Returns a DataFrame with industry output, emissions, tax revenue, and effective carbon tax rate.
    """
    
    q_j = IO_data['q_j']
    # set label for emission column
    if emission_column == 'CO2e Emissions (mt)':
        emission_column_alt = 'Carbon dioxide equivalent, in Tonnes'
    else:
        emission_column_alt = 'Carbon dioxide, in Tonnes'
    # Prepare tax overview table
    df_emissions = df_emissions.loc[(countries, sectors), :]
    q_j = q_j.loc[(countries, sectors)]
    tax_overview = pd.concat([df_emissions[emission_column_alt], q_j], axis=1)
    tax_overview[emission_column] = tax_overview[emission_column_alt] / 1e6  # Convert to mt
    tax_overview = tax_overview.drop(columns=[emission_column_alt])
    tax_overview.columns = ['Industry Output (mEUR)', emission_column]

    # Calculate tax revenue
    tax_overview['Tax Revenue (mEUR)'] = tax_overview[emission_column] * tax_rate

    # Calculate effective carbon tax rate as a percentage
    industry_output = tax_overview['Industry Output (mEUR)'].replace(0, np.nan)
    tax_overview['Effective Carbon Tax Rate (%)'] = (tax_overview['Tax Revenue (mEUR)'] / industry_output) * 100

    # sort by effective carbon tax rate
    tax_overview = tax_overview.sort_values(by='Effective Carbon Tax Rate (%)', ascending=False)
    return tax_overview

def CalculateTaxOverviewEU(
    tax_rate, 
    coverage_data=coverage_data_prepared, 
    countries=slice(None), 
    sectors=slice(None), 
    emission_column=emission_column, 
    IO_data=IO_data_EU, 
    df_emissions=df_emissions):
    
    """
    Calculate tax overview for the EU region. 
    Returns a DataFrame with industry output, emissions, existing carbon taxes, tax revenue, and effective carbon tax rate.
    """
    
    q_j = IO_data['q_j']
    # Prepare tax overview table
    df_emissions = df_emissions.loc[(countries, sectors), :]
    q_j = q_j.loc[(countries, sectors)]
    tax_overview = pd.concat([df_emissions[['Carbon dioxide equivalent, in Tonnes', 'Carbon dioxide, in Tonnes']], q_j], axis=1)
    tax_overview['CO2e Emissions (mt)'] = tax_overview['Carbon dioxide equivalent, in Tonnes'] / 1e6  # Convert to mt
    tax_overview = tax_overview.drop(columns=['Carbon dioxide equivalent, in Tonnes'])
    tax_overview['CO2 Emissions (mt)'] = tax_overview['Carbon dioxide, in Tonnes'] / 1e6  # Convert to mt
    tax_overview = tax_overview.drop(columns=['Carbon dioxide, in Tonnes'])
    
    tax_overview.columns = ['Industry Output (mEUR)', 'CO2e Emissions (mt)', 'CO2 Emissions (mt)']
    tax_overview = pd.concat([tax_overview, coverage_data], axis=1, join='inner')
    # Ensure prices are not above tax_rate
    tax_overview['Price_tCO2e_EUR'] = tax_overview['Price_tCO2e_EUR'].combine(
        pd.Series(tax_rate, index=tax_overview.index), 
        func=min
    )
    tax_overview = tax_overview.rename(columns={'Price_tCO2e_EUR': 'Existing Carbon Tax (EUR/tCO2)'})

    # Calculate tax revenue
    tax_revenue_base = tax_overview[emission_column] * tax_rate
    tax_substract = tax_overview['CO2 Emissions (mt)'] * tax_overview['Existing Carbon Tax (EUR/tCO2)'] # only CO2 taxed in existing taxes
    tax_overview['Tax Revenue (mEUR)'] = tax_revenue_base - tax_substract

    # Calculate effective carbon tax rate as a percentage
    industry_output = tax_overview['Industry Output (mEUR)'].replace(0, np.nan)
    tax_overview['Effective Carbon Tax Rate (%)'] = (tax_overview['Tax Revenue (mEUR)'] / industry_output) * 100

    # sort by effective carbon tax rate
    tax_overview = tax_overview.sort_values(by='Effective Carbon Tax Rate (%)', ascending=False)
    return tax_overview

def CalculateTaxEffects(
    tax_overview, 
    etha_j=etha_j, 
    IO_data=IO_data_DE):
    
    """ 
    Calculate the effects of the carbon tax on the economy.
    Returns a DataFrame with percentage change in prices and output for each sector, as well as output before and after the tax.
    """
    
    l_j=IO_data['l_j']
    q_j=IO_data['q_j']
    v_j=IO_data['v_j']
    L_ij=IO_data['L_ij']
    x_i=IO_data['x_i']

    # Calculate delta l_j
    l_0 = l_j.copy()  # Original primary input coefficients
    t_j = tax_overview['Tax Revenue (mEUR)']
    t_j.index.names = l_j.index.names
    t_j.name = 'Tax Revenue (t_j), in mEUR'

    # mask = q_j.index != 'U'
    # l_1 = (v_j + t_j)[mask] / (q_j + t_j)[mask]
    l_1 = (v_j + t_j) / (q_j + t_j)
    # l_1 = pd.concat([l_1, pd.Series(1.0, index=['U'])])
    l_1 = l_1.fillna(1.0)  # Fill NaN values (for sectors with zero output) with 1.0
    # 1.0 is used here as a neutral value, assuming no change in primary input
    l_1.index.names = l_j.index.names
    l_1.name = 'Primary Input Coefficients after Tax (l_1), in mEUR'

    delta_l_j = l_1 - l_0
    delta_l_j.name = 'Change in Primary Input Coefficients (delta l_j), in mEUR'

    delta_p_j = delta_l_j.dot(L_ij)
    delta_p_j.name = 'Change in Primary Input Coefficients after Tax (delta p_j)'

    perc_p_j = delta_p_j * 100
    perc_p_j.name = 'Percentage Change in Prices (delta p_j), in %'
    perc_p_j_sorted = perc_p_j.sort_values(ascending=False)

    q_1 = x_i * etha_j * delta_p_j + q_j

    delta_q_j = q_1 - q_j
    delta_q_j.name = 'Change in Output (delta q_j), in mEUR'

    perc_q_j = (delta_q_j / q_j) * 100
    # perc_q_j = pd.concat([perc_q_j, pd.Series(0.0, index=['U'])])
    perc_q_j = perc_q_j.replace([np.inf, -np.inf], np.nan).fillna(0)  # Replace inf values (for sectors with zero output) with 0
    perc_q_j.name = 'Percentage Change in Output (delta q_j), in %'
    perc_q_j_sorted = perc_q_j.sort_values(ascending=True)
    # Combine the two vectors into a DataFrame
    results_df = pd.DataFrame({
        'Percentage Change in Prices (%)': perc_p_j_sorted,
        'Percentage Change in Output (%)': perc_q_j_sorted,
        'Output before Change (mEUR)': q_j,
        'Output after Change (mEUR)': q_1
    })
    return results_df

def CalculateTaxOverviewAfterModel(
    tax_rate, 
    pre_tax_overview,
    emission_column=emission_column,
    etha_j=etha_j,
    IO_data=IO_data_DE
):
    '''
    Calculate the tax overview after the implementation of a GHG tax and corresponding model adjustment.
    Returns a DataFrame with updated industry output, emissions, tax revenue, and effective carbon tax rate.
    '''
    
    # prepare emission intensity
    emission_intensity_df, intensity_label = CalculateEmissionIntensities(tax_overview=pre_tax_overview, emission_column=emission_column)
    results_df = CalculateTaxEffects(tax_overview=pre_tax_overview, etha_j=etha_j, IO_data=IO_data)
    # Calculate post-tax overview after model adjustment
    post_tax_overview = pd.DataFrame(index=pre_tax_overview.index, columns=pre_tax_overview.columns)
    post_tax_overview['Industry Output (mEUR)'] = results_df['Output after Change (mEUR)']
    post_tax_overview[emission_column] = emission_intensity_df[intensity_label] * post_tax_overview['Industry Output (mEUR)']
    post_tax_overview['Tax Revenue (mEUR)'] = post_tax_overview[emission_column] * tax_rate
    # Calculate effective carbon tax rate as a percentage
    post_industry_output = post_tax_overview['Industry Output (mEUR)'].replace(0, np.nan)
    post_tax_overview['Effective Carbon Tax Rate (%)'] = (post_tax_overview['Tax Revenue (mEUR)'] / post_industry_output) * 100
    return post_tax_overview

def CalculateTaxOverviewWithEmissionReduction(
    tax_rate, 
    df_emissions=df_emissions,
    emission_reduction=emission_reduction, 
    countries=['DE'], 
    red_sector=slice(None), 
    emission_column=emission_column,
    etha_j=etha_j,
    IO_data=IO_data_DE
):
    '''
    Calculate the tax overview after the implementation of a GHG tax as well as sectoral emission reductions and corresponding model adjustment.
    Returns a DataFrame with updated industry output, emissions, tax revenue, and effective carbon tax rate, reflecting the emission reduction
    and a DataFrame before model adjustment with reduced emissions.
    '''
    # Apply emission reduction
    df_emissions_red = df_emissions.copy()
    df_emissions_red.loc[(countries, red_sector), :] *= (1 - emission_reduction)
    # Calculate initial tax overview with reduced emissions
    pre_tax_overview = CalculateTaxOverview(tax_rate, countries=countries, sectors=slice(None), emission_column=emission_column, IO_data=IO_data, df_emissions=df_emissions_red)
    # Calculate post-tax overview after model adjustment with reduced emissions
    post_tax_overview = CalculateTaxOverviewAfterModel(
        tax_rate=tax_rate,
        pre_tax_overview=pre_tax_overview,
        emission_column=emission_column,
        etha_j=etha_j,
        IO_data=IO_data
    )

    return pre_tax_overview, post_tax_overview

def CalculateCombinedResults(
    tax_rates=tax_rates,
    countries=['DE'],
    sectors=slice(None),
    emission_column=emission_column,
    etha_j=etha_j,
    IO_data=IO_data_DE,
    df_emissions=df_emissions,
    coverage_data_prepared=coverage_data_prepared,
    countries_type='DE' #'DE' for Germany, 'EU' for EU-wide (uses appropriate CalculateTaxOverview function)
):
    """
    Returns:
        1. DataFrame with combined CalculateTaxEffects-results for a list of tax rates.
        2. DataFrame with combined tax_overviews for a list of tax rates.
        3. first DataFrame sorted by price changes for highest tax rate.
        4. first DataFrame sorted by output changes for highest tax rate.
    """
    results_list = []
    tax_overview_list = []

    # Loop over tax rates and calculate results
    for rate in tax_rates:
        if countries_type == 'EU':
            tax_overview = CalculateTaxOverviewEU(
                rate,
                coverage_data=coverage_data_prepared,
                countries=countries,
                sectors=sectors,
                emission_column=emission_column,
                IO_data=IO_data,
                df_emissions=df_emissions
            )
        else:
            tax_overview = CalculateTaxOverview(
                rate,
                countries=countries,
                sectors=sectors,
                emission_column=emission_column,
                IO_data=IO_data,
                df_emissions=df_emissions
            )
        results_df = CalculateTaxEffects(tax_overview, etha_j, IO_data=IO_data)
        results_list.append(results_df)
        tax_overview_list.append(tax_overview)

    # Combine all results into a single DataFrame with MultiIndex for sector codes/names
    combined_results = pd.concat(results_list, keys=tax_rates, axis=1)
    combined_results = combined_results.apply(pd.to_numeric)
    combined_results.columns.names = ['Tax Rate (EUR/tCO2)', 'Metric']

    # Combine tax_overviews
    combined_tax_overview = pd.concat(tax_overview_list, keys=tax_rates, axis=1)
    combined_tax_overview.columns.names = ['Tax Rate (EUR/tCO2)', 'Metric']

    # Create sorted DataFrames
    combined_results_price = combined_results.sort_values((tax_rates[-1], 'Percentage Change in Prices (%)'), ascending=False)
    combined_results_output = combined_results.sort_values((tax_rates[-1], 'Percentage Change in Output (%)'), ascending=True)

    return combined_results, combined_tax_overview, combined_results_price, combined_results_output

def CalculateCoverageComparison(
    taxed_sector_scaling=1.0,
    partial_taxed_sector_scaling=1.0,
    coverage_data=coverage_data,
    df_emissions=df_emissions,
    coverage_source=coverage_source
):
    """
    Returns a DataFrame with CO2 coverage in the model and compares it with source data from World Bank.
    """
    
    # Calculate coverage for each country
    coverage_results = []
    for country_code in coverage_data.index.get_level_values(0).unique():
        co2_emissions_country, total_co2_emissions_country, taxed_sectors_list, partial_taxed_sectors_list = GetCountryCO2Coverage(
            country_code, coverage_data=coverage_data, df_emissions=df_emissions
        )
        coverage = (
            co2_emissions_country.loc[taxed_sectors_list].sum() * taxed_sector_scaling +
            co2_emissions_country.loc[partial_taxed_sectors_list].sum() * partial_taxed_sector_scaling
        ) / total_co2_emissions_country
        coverage_results.append({
            'Country Code': country_code,
            'Coverage in Model (%)': coverage * 100,
        })

    coverage_df = pd.DataFrame(coverage_results).set_index('Country Code')

    # Map country codes to country names
    coverage_df.index = coverage_df.index.map(country_name_dict)
    coverage_df.index.name = 'Country Name'

    # Compare with source data from World Bank
    coverage_source_local = coverage_source.set_index('Country Code')
    coverage_source_local = coverage_source_local.rename(columns={'Emissions Covered (%)': 'Coverage Source World Bank (%)'})
    coverage_source_local = coverage_source_local[['Coverage Source World Bank (%)']] * 100  # convert to percentage

    # Map index to country names for joining
    coverage_source_local.index = coverage_source_local.index.map(country_name_dict)
    coverage_source_local.index.name = 'Country Name'

    coverage_comparison = coverage_df.join(coverage_source_local, how='left')
    coverage_comparison['Difference (%)'] = coverage_comparison['Coverage in Model (%)'] - coverage_comparison['Coverage Source World Bank (%)']
    coverage_comparison = coverage_comparison.sort_values('Difference (%)', ascending=False)
    coverage_comparison = coverage_comparison.round(2)
    return coverage_comparison

def CalculateTotalEmissionSavings(
    tax_rates=tax_rates, 
    emission_reduction=emission_reduction, 
    emission_column=emission_column, 
    df_emissions=df_emissions, 
    etha_j=etha_j, 
    IO_data=IO_data_DE
):
    '''
    Returns a DataFrame with total emission savings for different tax rates and abatement scenarios.
    '''
    emissions_frame = pd.DataFrame(index=tax_rates, columns=['2019_base', 'modeled_emissions'] + IO_data['q_j'].index.get_level_values(1).to_list() + ['all_reduced'])
    for rate in tax_rates:
        pre_tax_overview = CalculateTaxOverview(tax_rate=rate, countries=['DE'], sectors=slice(None), emission_column=emission_column, IO_data=IO_data_DE, df_emissions=df_emissions)
        emissions_frame.loc[rate, '2019_base'] = pre_tax_overview[emission_column].sum()  # in Mt
        
        post_tax_overview = CalculateTaxOverviewAfterModel(
                tax_rate=rate, 
                pre_tax_overview = pre_tax_overview,
                emission_column=emission_column,
                etha_j=etha_j,
                IO_data=IO_data)
        emissions_frame.loc[rate, 'modeled_emissions'] = post_tax_overview[emission_column].sum()  # in Mt

        for sector in IO_data['q_j'].index:
            _, reduced_tax_overview = CalculateTaxOverviewWithEmissionReduction(
                tax_rate=rate,
                df_emissions=df_emissions,
                emission_reduction=emission_reduction,
                countries=['DE'],
                red_sector=[sector[1]],
                emission_column=emission_column,
                etha_j=etha_j,
                IO_data=IO_data
            )
            emissions_frame.loc[rate, sector[1]] = reduced_tax_overview[emission_column].sum()  # in Mt

        _, all_reduced_tax_overview = CalculateTaxOverviewWithEmissionReduction(
            tax_rate=rate,
            df_emissions=df_emissions,
            emission_reduction=emission_reduction,
            countries=['DE'],
            red_sector=slice(None),
            emission_column=emission_column,
            etha_j=etha_j,
            IO_data=IO_data
        )

        emissions_frame.loc[rate, 'all_reduced'] = all_reduced_tax_overview[emission_column].sum()  # in Mt


    return emissions_frame

def CalculateTotalOutputChanges(
    tax_rates=tax_rates, 
    emission_reduction=emission_reduction, 
    emission_column=emission_column, 
    output_column='Industry Output (mEUR)',
    df_emissions=df_emissions, 
    etha_j=etha_j, 
    IO_data=IO_data_DE
):
    '''
    Returns a DataFrame with total output changes for different tax rates and abatement scenarios.
    '''
    output_frame = pd.DataFrame(index=tax_rates, columns=['2019_base', 'modeled_emissions'] + IO_data['q_j'].index.get_level_values(1).to_list() + ['all_reduced'])
    for rate in tax_rates:
        pre_tax_overview = CalculateTaxOverview(tax_rate=rate, countries=['DE'], sectors=slice(None), emission_column=emission_column, IO_data=IO_data_DE, df_emissions=df_emissions)
        output_frame.loc[rate, '2019_base'] = pre_tax_overview[output_column].sum()  # in mEUR
        
        post_tax_overview = CalculateTaxOverviewAfterModel(
                tax_rate=rate, 
                pre_tax_overview = pre_tax_overview,
                emission_column=emission_column,
                etha_j=etha_j,
                IO_data=IO_data)
        output_frame.loc[rate, 'modeled_emissions'] = post_tax_overview[output_column].sum()  # in mEUR

        for sector in IO_data['q_j'].index:
            _, reduced_tax_overview = CalculateTaxOverviewWithEmissionReduction(
                tax_rate=rate,
                df_emissions=df_emissions,
                emission_reduction=emission_reduction,
                countries=['DE'],
                red_sector=[sector[1]],
                emission_column=emission_column,
                etha_j=etha_j,
                IO_data=IO_data
            )
            output_frame.loc[rate, sector[1]] = reduced_tax_overview[output_column].sum()  # in mEUR

        _, all_reduced_tax_overview = CalculateTaxOverviewWithEmissionReduction(
            tax_rate=rate,
            df_emissions=df_emissions,
            emission_reduction=emission_reduction,
            countries=['DE'],
            red_sector=slice(None),
            emission_column=emission_column,
            etha_j=etha_j,
            IO_data=IO_data
        )

        output_frame.loc[rate, 'all_reduced'] = all_reduced_tax_overview[output_column].sum()  # in mEUR

    #change to bEUR
    output_frame = output_frame / 1e3

    return output_frame

def CalculateAmortizationCostsAllSectors(
    sectors=IO_data_DE['q_j'].index,
    tax_rates=tax_rates,
    interest_rate=interest_rate,
    emission_reduction=emission_reduction,
    amortization_times_range=amortization_times_range,
    emission_column=emission_column,
    etha_j=etha_j,
    IO_data=IO_data_DE,
    df_emissions=df_emissions
):
    """
    Calculate Present Value (P) of avoided tax payments for all sectors and tax rates over a range of amortization times.
    Returns a DataFrame with amortization costs.
    """
    # Prepare MultiIndex columns: (tax_rate, amortization_time)
    columns = pd.MultiIndex.from_product([tax_rates, amortization_times_range], names=['Tax Rate', 'Years'])
    data = []
    for sector in sectors:
        row = []
        for rate in tax_rates:
            for n in amortization_times_range:
                cost = AmortizationCost(sector, rate, n, interest_rate, emission_reduction, emission_column, df_emissions, etha_j, IO_data)
                row.append(cost)
        data.append(row)
    amortization_costs = pd.DataFrame(data, index=sectors, columns=columns)
    return amortization_costs

##### Auxiliary Functions #####

def CalculateEmissionIntensities(
    tax_overview, 
    emission_column=emission_column
):
    """
    Calculate emission intensities using an existing tax_overview DataFrame.
    Assumes tax_overview contains columns for industry output and emissions.
    """
    
    if 'CO2e' in emission_column:
        intensity_label = 'Emission Intensity (mtCO2e/mEUR)'
    else:
        intensity_label = 'Emission Intensity (mtCO2/mEUR)'

    # Ensure required columns exist
    if emission_column not in tax_overview.columns or 'Industry Output (mEUR)' not in tax_overview.columns:
        raise ValueError("tax_overview must contain columns for emissions and industry output.")

    industry_output = tax_overview['Industry Output (mEUR)'].replace(0, np.nan)
    emission_intensity = tax_overview[emission_column] / industry_output

    emission_intensity_df = tax_overview.copy()
    emission_intensity_df[intensity_label] = emission_intensity

    return emission_intensity_df, intensity_label

def AmortizationCost(
    sector, 
    tax_rate, 
    amortization_time_sector, 
    interest_rate = interest_rate, 
    emission_reduction=emission_reduction,
    emission_column=emission_column,
    df_emissions=df_emissions,
    etha_j=etha_j,
    IO_data=IO_data_DE
):
    '''
    Calculate the Present Value (P) of avoided tax payments for a specific sector and tax rate.
    '''
    countries = [sector[0]]
    # prepare tax overview before model adjustment
    ex_ante_tax_overview = CalculateTaxOverview(tax_rate=tax_rate, countries=countries, sectors=slice(None), emission_column=emission_column, IO_data=IO_data, df_emissions=df_emissions)

    # prepare tax overview after model adjustment before emission reduction
    pre_tax_overview = CalculateTaxOverviewAfterModel(
        tax_rate=tax_rate, 
        pre_tax_overview=ex_ante_tax_overview,
        emission_column=emission_column,
        etha_j=etha_j,
        IO_data=IO_data)
    # prepare tax overview after model adjustment with emission reduction
    _, post_tax_overview = CalculateTaxOverviewWithEmissionReduction(
        tax_rate=tax_rate,
        df_emissions=df_emissions,
        emission_reduction=emission_reduction,
        countries=countries,
        red_sector=[sector[1]],
        emission_column=emission_column,
        etha_j=etha_j,
        IO_data=IO_data)

    pre_tax_revenue = pre_tax_overview.loc[sector, 'Tax Revenue (mEUR)']
    post_tax_revenue = post_tax_overview.loc[sector, 'Tax Revenue (mEUR)']
    if pre_tax_revenue <= post_tax_revenue:
        return 0.0
    n = amortization_time_sector
    r = interest_rate
    A = pre_tax_revenue - post_tax_revenue  # Annual savings in mEUR
    if r == 0:
        P = A * n
        return P
    # formula for amortization cost (initial investment P given annual savings A, interest rate r, and amortization time n)
    P = A * (1 - (1 + r) ** -n) / r
    return P

def GetTopSectorsByPriceChange(
    combined_results,
    tax_rate=max(tax_rates),
    exclude_sector=None,
    top_n=15
):
    """
    Returns the top n sectors by price change for the highest tax rate, excluding the specified sector.
    """
    price_changes = combined_results[tax_rate]['Percentage Change in Prices (%)']
    top_sectors = price_changes.nlargest(top_n).index  # Get top n sectors
    top_sectors = [sector for sector in top_sectors if sector != exclude_sector]  # Exclude specified sector
    return top_sectors

def GetCountryCO2Coverage(
    country_code, 
    coverage_data=coverage_data, 
    df_emissions=df_emissions
):
    """
    Returns CO2 emissions, total CO2 emissions, and lists of taxed and partially taxed sectors for a given country.
    Based on existing CO2 taxes in the EU27 as per coverage_data.
    """
    
    co2_emissions_country = df_emissions.loc[country_code, 'Carbon dioxide, in Tonnes']
    total_co2_emissions_country = co2_emissions_country.sum()
    coverage_data_country = coverage_data.loc[country_code].reset_index()
    taxed_sectors_list = coverage_data_country.loc[coverage_data_country.loc[:, 'Status_CO2'] == 'Y'].loc[:, 'NACE Code'].to_list()
    partial_taxed_sectors_list = coverage_data_country.loc[coverage_data_country.loc[:, 'Status_CO2'] == 'Partial'].loc[:, 'NACE Code'].to_list()
    return co2_emissions_country, total_co2_emissions_country, taxed_sectors_list, partial_taxed_sectors_list

##### Define Plotting Functions #####
def PlotCombinedPriceOutput(combined_results, top_sectors, tax_rates=tax_rates, num_sectors=15, save_str='combined_price_output_point_top.png'):
    '''
    Plot combined price and output changes for top sectors under Germany carbon tax scenarios.
    '''

    # save directory
    save_dir = 'Result Figures'

    # Get the top sectors with the biggest price changes for the highest tax rate
    top_sector_labels = [f"{sector}" for country, sector in top_sectors]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define point size and font sizes
    point_size = 100 
    labelsize = 16
    ticksize = 14
    legendsize = 14

    all_values = []
    for i, rate in enumerate(tax_rates):
        price_changes = combined_results[rate]['Percentage Change in Prices (%)'].loc[top_sectors]
        output_changes = combined_results[rate]['Percentage Change in Output (%)'].loc[top_sectors]
        all_values.extend(price_changes.values)
        all_values.extend(output_changes.values)

        # Plot price changes (upwards)
        ax.scatter(top_sector_labels, price_changes.values,
                    s=point_size, marker='o',
                    label=f'Price Change {rate} €/tCO2e', color=f'C{i}')
        # Plot output changes (downwards)
        ax.scatter(top_sector_labels, output_changes.values,
                    s=point_size, marker='D',
                    label=f'Output Change {rate} €/tCO2e', color=f'C{i}')

    # Set y-axis ticks to symmetric buckets around zero
    y_min = min(all_values)
    y_max = max(all_values)
    # Make symmetric around zero for better comparison
    abs_max = max(abs(y_min), abs(y_max))
    y_lim = (y_min * 1.4, y_max * 1.1)
    ax.set_ylim(y_lim)
    # Set bucket size (step)
    step = 2
    ax.set_yticks(np.arange(int(y_lim[0]), int(y_lim[1]) + step, step))

    # Increase label sizes
    ax.set_xlabel('Sector Code', fontsize=labelsize)
    ax.set_ylabel('Change in Prices/Output (%)', fontsize=labelsize)
    # Increase tick label size
    plt.xticks(rotation=45, ha='right', fontsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.tick_params(axis='x', labelsize=ticksize)

    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8)

    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate legend entries
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=legendsize)  # larger legend font

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotTopPriceChangesWithEU(combined_results, combined_results_EU_onlyDE, top_sectors, tax_rates=tax_rates, num_sectors=15, save_str='top_price_changes_DE_vs_EU.png'):
    """
    Plots top price changes for Germany and overlays EU data for the same sectors and tax rates.
    """
    # save directory
    save_dir = 'Result Figures'

    # Get top sectors by price change for Germany
    top_sector_labels = [f"{sector}" for country, sector in top_sectors]

    # Define font sizes
    labelsize = 16
    ticksize = 14
    legendsize = 14

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, rate in enumerate(tax_rates):
        # Germany data
        price_changes_de = combined_results[rate]['Percentage Change in Prices (%)'].loc[top_sectors]
        ax.scatter(
            top_sector_labels,
            price_changes_de.values,
            s=120,
            label=f'DE {rate} €/tCO2e',
            color=f'C{i}',
            marker='o',
            alpha=1
        )
        # EU data (Germany sectors only)
        price_changes_eu = combined_results_EU_onlyDE[rate]['Percentage Change in Prices (%)'].loc[top_sectors]
        ax.scatter(
            top_sector_labels,
            price_changes_eu.values,
            s=120,
            label=f'EU {rate} €/tCO2e',
            color=f'C{i}',
            marker='o',
            alpha=0.4
        )

    # Increase label and tick sizes
    ax.set_xlabel('Sector Code', fontsize=labelsize)
    ax.set_ylabel('Change in Prices (%)', fontsize=labelsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)

    # Legend with larger font size
    ax.legend(fontsize=legendsize)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotTopOutputChangesWithEU(combined_results, combined_results_EU_onlyDE, top_sectors, tax_rates=tax_rates, num_sectors=15, save_str='top_output_changes_DE_vs_EU.png'):
    """
    Plots top output changes for Germany and overlays EU data for the same sectors and tax rates.
    """
    # save directory
    save_dir = 'Result Figures'

    # Get top sectors by output change for Germany
    top_sector_labels = [f"{sector}" for country, sector in top_sectors]

    # Define font sizes
    labelsize = 16
    ticksize = 14
    legendsize = 14
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for rate in tax_rates:
        # Germany data
        output_changes_de = combined_results[rate]['Percentage Change in Output (%)'].loc[top_sectors]
        ax.scatter(
            top_sector_labels,
            output_changes_de.values,
            s=120,
            label=f'DE {rate} €/tCO2e',
            color=f'C{tax_rates.index(rate)}',
            marker='D',
            alpha=1
        )
        # EU data (Germany sectors only)
        output_changes_eu = combined_results_EU_onlyDE[rate]['Percentage Change in Output (%)'].loc[top_sectors]
        ax.scatter(
            top_sector_labels,
            output_changes_eu.values,
            s=120,
            label=f'EU {rate} €/tCO2e',
            color=f'C{tax_rates.index(rate)}',
            marker='D',
            alpha=0.4
        )
        
    # Increase label and tick sizes
    ax.set_xlabel('Country-Sector Code', fontsize=labelsize)
    ax.set_ylabel('Change in Output (%)', fontsize=labelsize)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=ticksize)
    ax.tick_params(axis='y', labelsize=ticksize)
    
    # Legend with larger font size
    ax.legend(fontsize=legendsize)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotAmortizationCostsSingleSector(amortization_costs_df, sector=('DE', 'D35'), tax_rates=tax_rates, amortization_times_range=amortization_times_range, save_str=f'amortization_costs_DE-D35.png'):
    """
    Plot Present Value (P) of avoided tax payments for a given sector over a range of amortization times and tax rates,
    using amortization_costs_df (MultiIndex columns: (tax_rate, years)).
    """
    # save directory
    save_dir = 'Result Figures'
    
    # Define font sizes
    labelsize = 16
    ticksize = 14
    legendsize = 14

    plt.figure(figsize=(10, 6))
    
    for rate in tax_rates:
        costs = amortization_costs_df.loc[sector, rate]
        plt.plot(amortization_times_range, costs.values, marker='o', label=f'Tax Rate: {rate} €/tCO2e')

    plt.xlabel('Amortization Time (years)', fontsize=labelsize)
    plt.ylabel('Present Value (mEUR)', fontsize=labelsize)
    plt.xticks(rotation=45, ha='right', fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(fontsize=legendsize)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotAmortizationCostsCustomSectors(amortization_costs_df, sectors, tax_rates=tax_rates, amortization_times_range=amortization_times_range, save_str='amortization_costs_custom_layout.png'):
    """
    Plot Present Value (P) of avoided tax payments for a custom list of sectors and tax rates,
    using amortization_costs_df for the new output format (MultiIndex columns: (tax_rate, years)).
    Layout: square (2x2): top left, top right, bottom left = graphs; bottom right = legend.
    """
    if len(tax_rates) != 3:
        raise ValueError("This function requires exactly 3 tax rates for the custom layout.")

    # save directory
    save_dir = 'Result Figures'
    
    # Define font sizes
    labelsize = 18
    ticksize = 15
    legendsize = 15
    titlesize = 18

    amortization_costs_df = amortization_costs_df.loc[sectors]
    # Flatten all costs for axis range calculation
    all_costs = amortization_costs_df.values.flatten()
    y_min = np.nanmin(all_costs)
    y_max = np.nanmax(all_costs)
    # Add padding to y-axis limits for better visualization
    y_range = y_max - y_min
    y_pad = y_range * 0.08 if y_range > 0 else 1.0
    y_min_padded = y_min - y_pad
    y_max_padded = y_max + y_pad

    # Square layout: 2x2, each subplot same size
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes = axes.flatten()
    legend_ax = axes[3]
    legend_ax.axis('off')

    legend_handles = []
    legend_labels = []
    for idx, rate in enumerate(tax_rates):
        ax = axes[idx]
        for sector in sectors:
            costs = amortization_costs_df.loc[sector, rate]
            line, = ax.plot(
                amortization_times_range,
                costs.values,
                marker='o',
                label=f'{sector[1]}'
            )
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(f'{sector[1]}')
        ax.set_xlabel('Amortization Time (years)', fontsize=labelsize)
        ax.set_ylabel('Present Value (mEUR)', fontsize=labelsize)
        ax.set_title(f'Tax Rate: {rate} €/tCO2e', fontsize=titlesize)
        ax.grid(True)
        
        
        ax.set_ylim(y_min_padded, y_max_padded)
        ax.tick_params(axis='both', labelsize=ticksize)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=ticksize)

    # Legend in bottom right
    legend_ax.legend(legend_handles, legend_labels, loc='center', fontsize=legendsize, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotTotalEmissionSavings(emissions_frame, sectors, tax_rates=tax_rates, save_str='total_emission_savings.png'):
    """
    Plots total emissions for different tax rates and abatement scenarios.
    """
    # save directory
    save_dir = 'Result Figures'
    
    scenarios = ['2019_base', 'modeled_emissions'] + [sector[1] for sector in sectors] + ['all_reduced']
    x = np.arange(len(scenarios))
    bar_width = 0.15

    # Define font sizes
    labelsize = 16
    ticksize = 14
    legendsize = 14
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_emissions = []
    for i, rate in enumerate(tax_rates):
        emissions = emissions_frame.loc[rate, scenarios].values
        all_emissions.extend(emissions)
        ax.bar(x + i * bar_width, emissions, width=bar_width, label=f'Tax Rate: {rate} €/tCO2e')

    # Set y-axis limits to relevant range (min/max with padding)
    min_em = min(all_emissions)
    max_em = max(all_emissions)
    y_pad = (max_em - min_em) * 0.25 if max_em > min_em else 1.0
    ax.set_ylim(min_em - y_pad, max_em + y_pad)

    ax.set_xticks(x + bar_width * (len(tax_rates) - 1) / 2)
    ax.set_xticklabels(['2019 data', 'After Tax'] + [sector[1] for sector in sectors] + ['All sectors'], rotation=45, ha='right', fontsize=ticksize)
    ax.set_ylabel('Total Emissions Germany (mt CO2e)', fontsize=labelsize)
    ax.set_xlabel('Scenario', fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.grid(True, axis='y')
    ax.legend(loc='lower left', fontsize=legendsize)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotTotalOutputChanges(output_frame, sectors, tax_rates=tax_rates, save_str='total_output_changes.png'):
    """
    Plots total industry output for different tax rates and abatement scenarios.
    """
    save_dir = 'Result Figures'
    
    scenarios = ['2019_base', 'modeled_emissions'] + [sector[1] for sector in sectors] + ['all_reduced']
    x = np.arange(len(scenarios))
    bar_width = 0.15

    # Define font sizes
    labelsize = 16
    ticksize = 14
    legendsize = 14
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_outputs = []
    for i, rate in enumerate(tax_rates):
        outputs = output_frame.loc[rate, scenarios].values
        all_outputs.extend(outputs)
        ax.bar(x + i * bar_width, outputs, width=bar_width, label=f'Tax Rate: {rate} €/tCO2e')

    # Set y-axis limits to relevant range (min/max with padding)
    min_out = min(all_outputs)
    max_out = max(all_outputs)
    y_pad = (max_out - min_out) * 0.2 if max_out > min_out else 1.0
    ax.set_ylim(min_out - y_pad, max_out + y_pad)

    ax.set_xticks(x + bar_width * (len(tax_rates) - 1) / 2)
    ax.set_xticklabels(['2019 data', 'After Tax'] + [sector[1] for sector in sectors] + ['All sectors'], rotation=45, ha='right', fontsize=ticksize)
    ax.set_ylabel('Total Industry Output Germany (bEUR)', fontsize=labelsize)
    ax.set_xlabel('Scenario', fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.grid(True, axis='y')
    ax.legend(loc='upper right', fontsize=legendsize)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotStackedEmissionsDE(df_emissions=df_emissions, save_str='total_emissions_DE.png'):
    """
    Plots a stacked bar chart of CO2, methane (CO2e), and nitrous oxide (CO2e) emissions for all german sectors.
    """
    save_dir = 'Result Figures'
    
    # Filter for Germany
    de_emissions = df_emissions.loc['DE']
    # Get sector codes
    sectors = de_emissions.index.tolist()
    # Extract emissions
    co2 = de_emissions['Carbon dioxide, in Tonnes'] / 1e6  # Convert to Mt
    methane = de_emissions['Methane (CO2 equivalent), in Tonnes'] / 1e6  # Convert to Mt
    nitrous = de_emissions['Nitrous oxide (CO2 equivalent), in Tonnes'] / 1e6  # Convert to Mt

    # Define font sizes
    labelsize = 18
    ticksize = 14
    ticksize_small = 10
    legendsize = 14

    # Prepare plot
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sectors))
    ax.bar(x, co2, label='Carbon Dioxide', color='#1f77b4')
    ax.bar(x, methane, bottom=co2, label='Methane (in CO2e)', color='#ff7f0e')
    ax.bar(x, nitrous, bottom=co2 + methane, label='Nitrous Oxide (in CO2e)', color='#2ca02c')

    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=45, ha='right', fontsize=ticksize_small)
    ax.set_ylabel('Emissions (mt CO2e)', fontsize=labelsize)
    ax.set_xlabel('Sector Code', fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.grid(True, axis='y')
    ax.legend(fontsize=legendsize)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

def PlotStackedEmissionIntensitiesDE(df_emissions=df_emissions, IO_data=IO_data_DE, save_str='total_emission_intensities_DE.png'):
    """
    Plots a stacked bar chart of emission intensities (CO2, methane (CO2e), nitrous oxide (CO2e)) for all german sectors.
    Emission intensity = emissions / output (mt CO2e per mEUR).
    """
    save_dir = 'Result Figures'

    # Filter for Germany
    de_emissions = df_emissions.loc['DE']
    # Get sector codes
    sectors = de_emissions.index.tolist()
    # Extract emissions (in tonnes)
    co2 = de_emissions['Carbon dioxide, in Tonnes'] / 1e6  # Mt
    methane = de_emissions['Methane (CO2 equivalent), in Tonnes'] / 1e6  # Mt
    nitrous = de_emissions['Nitrous oxide (CO2 equivalent), in Tonnes'] / 1e6  # Mt

    # Get industry output (mEUR)
    output = IO_data['q_j'].loc[('DE', slice(None))].values
    # Avoid division by zero
    output = np.where(output == 0, np.nan, output)

    # Calculate emission intensities (mt per mEUR)
    co2_intensity = co2.values / output
    methane_intensity = methane.values / output
    nitrous_intensity = nitrous.values / output

    # Define font sizes
    labelsize = 18
    ticksize = 14
    ticksize_small = 10
    legendsize = 14

    # Prepare plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(sectors))
    ax.bar(x, co2_intensity, label='Carbon Dioxide', color='#1f77b4')
    ax.bar(x, methane_intensity, bottom=co2_intensity, label='Methane (in CO2e)', color='#ff7f0e')
    ax.bar(x, nitrous_intensity, bottom=co2_intensity + methane_intensity, label='Nitrous Oxide (in CO2e)', color='#2ca02c')

    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=45, ha='right', fontsize=ticksize_small)
    ax.set_ylabel('Emission Intensity (tCO2e/€)', fontsize=labelsize)
    ax.set_xlabel('Sector Code', fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=ticksize)
    ax.grid(True, axis='y')
    ax.legend(fontsize=legendsize)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_str))
    plt.show()

###########################################################################
#%% 3: Calculate Carbon Tax Effects
###########################################################################

# Calculate combined results for Germany only
combined_results, combined_tax_overview, combined_results_price, combined_results_output = CalculateCombinedResults(
    tax_rates,
    countries=['DE'],
    sectors=slice(None),
    etha_j=etha_j,
    IO_data=IO_data_DE,
    df_emissions=df_emissions,
    coverage_data_prepared=coverage_data_prepared,
    countries_type='DE'
)

# Calculate combined results for EU (all countries)
combined_results_EU, combined_tax_overview_EU, combined_results_price_EU, combined_results_output_EU = CalculateCombinedResults(
    tax_rates,
    countries=slice(None),
    sectors=slice(None),
    etha_j=etha_j_EU,
    IO_data=IO_data_EU,
    df_emissions=df_emissions,
    coverage_data_prepared=coverage_data_prepared,
    countries_type='EU'
)
# Get EU results for Germany only
combined_results_EU_onlyDE = combined_results_EU.loc[('DE', slice(None)), :]
combined_results_price_EU_onlyDE = combined_results_price_EU.loc[('DE', slice(None)), :]
combined_results_output_EU_onlyDE = combined_results_output_EU.loc[('DE', slice(None)), :]
combined_tax_overview_EU_onlyDE = combined_tax_overview_EU.loc[('DE', slice(None)), :]


###########################################################################
#%% 4: Amortization Costs
###########################################################################

# Calculate amortization costs for all sectors and tax rates over a range of amortization times for Germany
amortization_costs_all_sectors = CalculateAmortizationCostsAllSectors(
    sectors=IO_data_DE['q_j'].index,
    tax_rates=tax_rates,
    interest_rate=interest_rate,
    emission_reduction=emission_reduction,
    amortization_times_range=amortization_times_range,
    emission_column=emission_column,
    etha_j=etha_j,
    IO_data=IO_data_DE,
    df_emissions=df_emissions
)

###########################################################################
#%% 5: Emission and Output Savings
###########################################################################

# total emission savings
total_emission_savings = CalculateTotalEmissionSavings(
    tax_rates=tax_rates, 
    emission_reduction=emission_reduction, 
    emission_column=emission_column, 
    df_emissions=df_emissions, 
    etha_j=etha_j, 
    IO_data=IO_data_DE
)

# total output changes
total_output_changes = CalculateTotalOutputChanges(
    tax_rates=tax_rates, 
    emission_reduction=emission_reduction, 
    emission_column=emission_column,
    output_column='Industry Output (mEUR)', 
    df_emissions=df_emissions, 
    etha_j=etha_j, 
    IO_data=IO_data_DE
)

# calculate relative total emissions of D35 and A01 sector in Germany
relative_emission_D35 = df_emissions.loc[('DE', 'D35'), 'Carbon dioxide equivalent, in Tonnes'] / df_emissions.loc['DE', 'Carbon dioxide equivalent, in Tonnes'].sum()
relative_emission_A01 = df_emissions.loc[('DE', 'A01'), 'Carbon dioxide equivalent, in Tonnes'] / df_emissions.loc['DE', 'Carbon dioxide equivalent, in Tonnes'].sum()


###########################################################################
#%% 6: Coverage comparison
###########################################################################

# Calculate coverage comparison
coverage_comparison = CalculateCoverageComparison(
    taxed_sector_scaling=1,
    partial_taxed_sector_scaling=1,
    coverage_data=coverage_data,
    df_emissions=df_emissions,
    coverage_source=coverage_source
)

###########################################################################
#%% 7: Create Plots
###########################################################################
### Plot total emissions stacked for Germany (Fig 5.1)
PlotStackedEmissionsDE(df_emissions=df_emissions, save_str='total_emissions_DE.png')

### Plot total emission intensities stacked for Germany (Fig 5.2)
PlotStackedEmissionIntensitiesDE(df_emissions=df_emissions, IO_data=IO_data_DE, save_str='total_emission_intensities_DE.png')


### Get Top 15 sectors by price change under German tax
top_sectors_pricechange = GetTopSectorsByPriceChange(combined_results=combined_results, tax_rate=max(tax_rates), top_n=15)   

### Price and Output Effects for Germany (Fig 5.3)
PlotCombinedPriceOutput(combined_results, top_sectors_pricechange, tax_rates, num_sectors=15, save_str='combined_price_output_DE.png')

### Comparison of Price and Output Effects Germany vs EU (Fig 5.4 and 5.5)
PlotTopPriceChangesWithEU(combined_results, combined_results_EU_onlyDE, top_sectors_pricechange, tax_rates, num_sectors=15, save_str='top_price_changes_DE_vs_EU.png')
PlotTopOutputChangesWithEU(combined_results, combined_results_EU_onlyDE, top_sectors_pricechange, tax_rates, num_sectors=15, save_str='top_output_changes_DE_vs_EU.png')

### Amortization
# exclude D35 as it has a seperate plot
top_sectors_pricechange_noD35 = GetTopSectorsByPriceChange(combined_results=combined_results, tax_rate=max(tax_rates), top_n=15, exclude_sector=('DE', 'D35'))
# other top sectors (Fig 5.6)
PlotAmortizationCostsCustomSectors(sectors=top_sectors_pricechange_noD35, tax_rates=tax_rates, amortization_costs_df=amortization_costs_all_sectors, amortization_times_range=amortization_times_range, save_str='amortization_costs_custom_topprice.png')
# only D35 Germany (Fig 5.7)
PlotAmortizationCostsSingleSector(amortization_costs_df=amortization_costs_all_sectors, sector=('DE', 'D35'), tax_rates=tax_rates, amortization_times_range=amortization_times_range, save_str='amortization_costs_DE-D35.png')

### Total emission savings (Fig 5.8)
PlotTotalEmissionSavings(emissions_frame=total_emission_savings, sectors=top_sectors_pricechange, tax_rates=tax_rates, save_str='total_emission_savings.png')

### Total output changes (Fig 5.9)
PlotTotalOutputChanges(output_frame=total_output_changes, sectors=top_sectors_pricechange, tax_rates=tax_rates, save_str='total_output_changes.png')


###########################################################################
#%% 8: Save Results and create Latex Tables
###########################################################################

### Germany Data
with pd.ExcelWriter('Result Tables/carbon_tax_results_DE.xlsx') as writer:
    combined_results.to_excel(writer, sheet_name='All Results')
    combined_results_price.to_excel(writer, sheet_name='Sorted by Price Change')
    combined_results_output.to_excel(writer, sheet_name='Sorted by Output Change')
    
# Latex table with reduced columns (Basis for Tab A.5)
combined_results.columns
columns_to_include = [( 65, 'Percentage Change in Prices (%)'),
                    ( 65, 'Percentage Change in Output (%)'),
                    ( 120, 'Percentage Change in Prices (%)'),
                    ( 120, 'Percentage Change in Output (%)'),
                    ( 171, 'Percentage Change in Prices (%)'),
                    ( 171, 'Percentage Change in Output (%)')]  
combined_results_reduced = combined_results.loc[:, columns_to_include]
combined_results_reduced.columns = [( 65, 'Change in Prices (%)'),
                                ( 65, 'Change in Output (%)'),
                                ( 120, 'Change in Prices (%)'),
                                ( 120, 'Change in Output (%)'),
                                ( 171, 'Change in Prices (%)'),
                                ( 171, 'Change in Output (%)')]
latex_str = combined_results_reduced.to_latex(escape=False, index=True, float_format="%.2f") 
latex_str = latex_str.replace('%', r'\%')
with open('Result Tables/carbon_tax_results_DE_reduced_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)
    
### EU Data
with pd.ExcelWriter('Result Tables/carbon_tax_results_EU.xlsx') as writer:
    combined_results_EU_onlyDE.to_excel(writer, sheet_name='All Results DE only')
    combined_results_price_EU_onlyDE.to_excel(writer, sheet_name='Sorted by Price Change DE only')
    combined_results_output_EU_onlyDE.to_excel(writer, sheet_name='Sorted by Output Change DE only')
    
# Latex table with reduced columns (Basis for Tab A.6)
combined_results_EU_onlyDE_reduced = combined_results_EU_onlyDE.loc[:, columns_to_include]
combined_results_EU_onlyDE_reduced.columns = [( 65, 'Change in Prices (%)'),
                                ( 65, 'Change in Output (%)'),
                                ( 120, 'Change in Prices (%)'),
                                ( 120, 'Change in Output (%)'),
                                ( 171, 'Change in Prices (%)'),
                                ( 171, 'Change in Output (%)')]
latex_str = combined_results_EU_onlyDE_reduced.to_latex(escape=False, index=True, float_format="%.2f")
latex_str = latex_str.replace('%', r'\%')
latex_str = latex_str.replace('_', r'\_')
with open('Result Tables/carbon_tax_results_EU_onlyDE_reduced_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)

### Tax overview
with pd.ExcelWriter('Result Tables/tax_overviews.xlsx') as writer:
    combined_tax_overview.to_excel(writer, sheet_name='Tax Overview DE')
    combined_tax_overview_EU_onlyDE.to_excel(writer, sheet_name='Tax Overview EU, DE only')
    
# Latex table with reduced columns (Basis for Tab A.4)
combined_tax_overview.columns
columns_to_include_overview = [( 65,            'Tax Revenue (mEUR)'),
                            ( 65, 'Effective Carbon Tax Rate (%)'),
                            (120,            'Tax Revenue (mEUR)'),
                            (120, 'Effective Carbon Tax Rate (%)'),
                            (171,            'Tax Revenue (mEUR)'),
                            (171, 'Effective Carbon Tax Rate (%)')]
combined_tax_overview_reduced = combined_tax_overview.loc[IO_data_DE['q_j'].index, columns_to_include_overview]
latex_str = combined_tax_overview_reduced.to_latex(escape=False, index=True, float_format="%.2f")
latex_str = latex_str.replace('%', r'\%')
latex_str = latex_str.replace('_', r'\_')
with open('Result Tables/tax_overview_DE_reduced_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)

### Amortization costs
with pd.ExcelWriter('Result Tables/amortization_costs_all_sectors_DE_mEUR.xlsx') as writer:
    amortization_costs_all_sectors.to_excel(writer, sheet_name='Amortization Costs')
    
# Latex table with reduced columns (Basis for Tab A.7)
amortization_costs_all_sectors.columns
amortization_costs_all_sectors_reduced = amortization_costs_all_sectors.loc[:, (slice(None), [10, 20])]
latex_str = amortization_costs_all_sectors_reduced.to_latex(escape=False, index=True, float_format="%.2f")
latex_str = latex_str.replace('%', r'\%')
latex_str = latex_str.replace('_', r'\_')
with open('Result Tables/amortization_costs_all_sectors_DE_mEUR_reduced_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)

### Total emission savings
with pd.ExcelWriter('Result Tables/total_emission_savings_DE_mtCO2e.xlsx') as writer:
    total_emission_savings.to_excel(writer, sheet_name='Total Emission Savings')

### Total output changes
with pd.ExcelWriter('Result Tables/total_output_changes_DE_bEUR.xlsx') as writer:
    total_output_changes.to_excel(writer, sheet_name='Total Output Changes')

### Total Output and emission changes combined latex table (Basis for Tab A.8)
total_output_emission_changes = pd.concat([total_emission_savings.T, total_output_changes.T], axis=1, keys=['Total Emission Savings (mt CO2e)', 'Total Output Changes (bEUR)'])
total_output_emission_changes.index = total_output_emission_changes.index.astype(str).str.replace('2019_base', '2019 data').str.replace('modeled_emissions', 'After Tax').str.replace('all_reduced', 'All sectors')
latex_str = total_output_emission_changes.to_latex(escape=False, index=True, float_format="%.2f")
latex_str = latex_str.replace('%', r'\%')
latex_str = latex_str.replace('_', r'\_')
with open('Result Tables/total_output_emission_changes_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)

### Coverage comparison
with pd.ExcelWriter('Result Tables/coverage_comparison.xlsx') as writer:
    coverage_comparison.to_excel(writer, sheet_name='Coverage Comparison')
    
# Save as LaTeX table (Basis for Tab 4.1)
latex_str = coverage_comparison.to_latex(escape=False, index=True, float_format="%.2f")
latex_str = latex_str.replace('%', r'\%')
with open('Result Tables/coverage_comparison_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)

### Emission data
with pd.ExcelWriter('Result Tables/emission_data.xlsx') as writer:
    df_emissions.to_excel(writer, sheet_name='Emission Data 2019')
    
### latex table for emission data Germany only (Basis for Tab A.3)
df_emissions_DE = df_emissions.loc[('DE', slice(None)), :]
df_emissions_DE.columns
df_emissions_DE_reduced = df_emissions_DE.loc[:, ['Carbon dioxide, in Tonnes',
                                                  'Methane (CO2 equivalent), in Tonnes',
                                                  'Nitrous oxide (CO2 equivalent), in Tonnes']]
df_emissions_DE_reduced['Total CO2e (Tonnes)'] = df_emissions_DE_reduced.sum(axis=1)
df_emissions_DE_reduced = pd.concat([df_emissions_DE_reduced, IO_data_DE['q_j']], axis=1)
df_emissions_DE_reduced.columns = ['CO2 (t)', 'Methane (tCO2e)', 'Nitrous Oxide (tCO2e)', 'Total CO2e (t)', 'Total Output (mEUR)']
df_emissions_DE_reduced['Total Emissions (mtCO2e)'] = df_emissions_DE_reduced['Total CO2e (t)'] / 1e6
# include emission intensity
df_emissions_DE_reduced['Emission Intensity (tCO2e/€)'] = df_emissions_DE_reduced['Total Emissions (mtCO2e)'] / (df_emissions_DE_reduced['Total Output (mEUR)'])
df_emissions_DE_reduced = df_emissions_DE_reduced[['Total Emissions (mtCO2e)', 'Total Output (mEUR)', 'Emission Intensity (tCO2e/€)']]
df_emissions_DE_reduced = df_emissions_DE_reduced.round({'Total Emissions (mtCO2e)': 2, 'Total Output (mEUR)': 2, 'Emission Intensity (tCO2e/€)': 6})

latex_str = df_emissions_DE_reduced.to_latex(escape=False, index=True)
latex_str = latex_str.replace('%', r'\%')
latex_str = latex_str.replace('_', r'\_')
with open('Result Tables/emission_data_DE_reduced_table.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)
