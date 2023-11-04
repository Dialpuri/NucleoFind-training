import generate_feature_lists
import generate_segmented_maps
import generate_test_train_data

def main(): 
    generate_segmented_maps.generate_map_files()
    generate_feature_lists.generate_help_files()
    generate_feature_lists.seed_c_alpha_positions()
    generate_test_train_data.combine_help_files()
    generate_test_train_data.generate_test_train_split()
    
if __name__ == "__main__":
    main()