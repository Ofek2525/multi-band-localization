import pandas as pd
import os


def filter_csv(input_filename: str, output_filename: str, columns_to_keep: list):
    # Load the original CSV file
    df = pd.read_csv(input_filename)

    # Filter rows where 'link state' is 1
    df_filtered = df[df['link state'] == 1]

    # Keep only the specified columns
    df_filtered = df_filtered[columns_to_keep]

    # Write to a new CSV file
    df_filtered.to_csv(output_filename, index=False)


def split_csv(filtered_input_filename: str, train_filename: str, test_filename: str):
    # Load the filtered CSV
    df = pd.read_csv(filtered_input_filename)

    # Reset index to ensure proper slicing
    df = df.reset_index(drop=True)

    # Every 10th row for test
    test_df = df.iloc[::10].reset_index(drop=True)

    # The rest for train
    train_df = df.drop(df.index[::10]).reset_index(drop=True)

    # Save to files
    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)


def check_test_files_are_same(file_type):
    filenames = [
        f'resources/raytracing/6000/LOS_bs1_6k_{file_type}.csv', 
        f'resources/raytracing/12000/LOS_bs1_12k_{file_type}.csv',
        f'resources/raytracing/18000/LOS_bs1_18k_{file_type}.csv', 
        f'resources/raytracing/24000/LOS_bs1_24k_{file_type}.csv'
    ]

    # Read CSV files
    dfs = [pd.read_csv(fname) for fname in filenames]

    # Find the minimum length to avoid index errors
    min_len = min(len(df) for df in dfs)

    mismatch_count = 0

    for i in range(min_len):
        current_rows = [df.iloc[i] for df in dfs]
        values = [tuple(row[col] for col in ['rx_x', 'rx_y', 'delay_1', 'aod_1']) for row in current_rows]

        if all(v == values[0] for v in values):
            continue  # All match, continue
        else:
            mismatch_count += 1
            print(f"\nMismatch at row {i}:")
            for j, val in enumerate(values):
                print(f"  File {j+1}, Row {i}: {val}")
            input("Press Enter to continue...")

    print(f"\nCheck complete. {min_len} rows checked. {mismatch_count} mismatches found.")


def split_csv_18k(file_type):
    # Load files
    dataset_bad = pd.read_csv("resources/raytracing/18000/LOS_bs1_18k.csv")
    train_ref = pd.read_csv(f"resources/raytracing/6000/LOS_bs1_6k_{file_type}.csv")

    # Clean and prepare output
    fixed_train_rows = []

    # Build a lookup for faster access
    dataset_lookup = dataset_bad.set_index(['rx_x', 'rx_y'])

    # Go through the training reference line by line
    for _, row in train_ref.iterrows():
        key = (row['rx_x'], row['rx_y'])
        if key in dataset_lookup.index:
            matched_row = dataset_lookup.loc[key]
            # If there's only one match, matched_row is a Series
            if isinstance(matched_row, pd.Series):
                full_row = matched_row.to_dict()
                full_row['rx_x'] = key[0]
                full_row['rx_y'] = key[1]
                fixed_train_rows.append(full_row)
            else:
                full_row = matched_row.iloc[0].to_dict()
                full_row['rx_x'] = key[0]
                full_row['rx_y'] = key[1]
                fixed_train_rows.append(full_row)
        else:
            print(f"No match found in original dataset for rx_x={key[0]}, rx_y={key[1]}")

    # Create DataFrame from collected rows
    fixed_train = pd.DataFrame(fixed_train_rows)

    # Ensure 'rx_x' and 'rx_y' are in columns 3 and 4
    cols = fixed_train.columns.tolist()

    # Remove 'rx_x' and 'rx_y' if they exist elsewhere
    cols = [col for col in cols if col not in ['rx_x', 'rx_y']]

    # Insert 'rx_x' and 'rx_y' at positions 2 and 3 (3rd and 4th columns)
    cols.insert(2, 'rx_x')
    cols.insert(3, 'rx_y')

    # Reorder columns
    fixed_train = fixed_train[cols]

    # Save to CSV
    fixed_train.to_csv(f"resources/raytracing/18000/LOS_bs1_18k_{file_type}.csv", index=False)
    print("fixed.csv created with", len(fixed_train), "rows")


def check_two_files_interactively(file1, file2):
    # Read both files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure both files have the same number of rows
    num_rows = min(len(df1), len(df2))
    print(f"Comparing {num_rows} rows...")

    for i in range(num_rows):
        row1 = df1.iloc[i]
        row2 = df2.iloc[i]

        val1 = tuple(row1[col] for col in ['rx_x', 'rx_y', 'delay_1', 'aod_1'])
        val2 = tuple(row2[col] for col in ['rx_x', 'rx_y', 'delay_1', 'aod_1'])

        if val1 != val2:
            print(f"\nMismatch at row {i}:")
            print(f"  File 1: {val1}")
            print(f"  File 2: {val2}")
            input("Press Enter to continue...")

    print("Check complete.")


if __name__ == "__main__":
    file_type = "train"
    # split_csv_18k(file_type)
    check_test_files_are_same(file_type)
    # check_two_files_interactively(
    #     f'resources/raytracing/6000/LOS_bs1_6k_{file_type}.csv',
    #     f'resources/raytracing/18000/LOS_bs1_18k_{file_type}.csv'
    # )

