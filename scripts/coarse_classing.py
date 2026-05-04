def coarse_class_variable(fine_summary_df, bin_mapping):
    """
    applies coarse classing to a fine classing summary table.
    
    returns:
    coarse calss summary with aggregated bins 
    """
    try:
# reverse mapping (fine bin to coarse label)
        reverse_map ={}
        for coarse_label, fine_bins in bin_mapping.items():
            for fine_bin in fine_bins:
                reverse_map[fine_bin] = coarse_label
    
        # add new column to fine summary with the coarse label
        fine_summary_df['Coarse_Bin'] = fine_summary_df['Bin'].astype(str).map(reverse_map)
        
# group by coarse bins and sum counts
        grouped = fine_summary_df.groupby('Coarse_Bin').agg({
            'count': 'sum',
            'Goods': 'sum',
            'Bads': 'sum'
        }).reset_index()
        
# calculate bad rate
        grouped['Bad rate(%)'] = (grouped['Bads'] / grouped['count'] * 100).round(2)
        
        return grouped[['Coarse_Bin', 'count', 'Goods', 'Bads', 'Bad rate(%)']]
    
    except Exception as e:
        print(f"Error in coarse classing: {e}")
        return None