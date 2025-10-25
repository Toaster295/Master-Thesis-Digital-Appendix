# Master-Thesis-Digital-Appendix
Contains raw data, Python scripts, and output files used to produce the results presented in my Master’s thesis.

# IMPORTANT NOTE:
The raw FIGARO input-output table is not uploaded because it is too large for GitHub!
You can download it here: 
  https://ec.europa.eu/eurostat/web/esa-supply-use-input-tables/database
Choose the following table in .tsv format: 
  "EU inter-country input-output table at basic prices, product by product (2018-2021) (naio_10_fcp_ip3)"
and replace the file in the following location (Check, that the name is the same):
  Figaro Data/estat_naio_10_fcp_ii3.tsv

# HOW TO RUN THE CODES
First, after downloading the FIGARO file as described above, run "Prepare_data.py". This will create the prepared data files in a new folder "Prepared Data/".
Afterwars, you can run "Calculate_results_and_plots.py". Carefull, this will overwrite the existing Excels, Latex Tables and Plots in "Result Figures/" and "Result Tables". 
