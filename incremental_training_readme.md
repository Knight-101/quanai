# Using Google Drive with Incremental Training

This guide explains how to use Google Drive for market data with the incremental training scripts.

## Setting Up

1. **Create your Google Drive file IDs JSON file**:

   Create a file called `drive_file_ids.json` with the Google Drive file IDs for your market data:

   ```json
   {
     "market_data/binance_BTCUSDT_5m.parquet": "your_btc_file_id_here",
     "market_data/binance_ETHUSDT_5m.parquet": "your_eth_file_id_here",
     "market_data/binance_SOLUSDT_5m.parquet": "your_sol_file_id_here",

     "perpetual/binance/5m/BTCUSDT.parquet": "your_btc_perp_file_id_here",
     "perpetual/binance/5m/ETHUSDT.parquet": "your_eth_perp_file_id_here",
     "perpetual/binance/5m/SOLUSDT.parquet": "your_sol_perp_file_id_here"
   }
   ```

2. **Get the file IDs from Google Drive**:
   - Upload your files to Google Drive
   - Right-click on each file → Share → Copy link
   - Extract the ID from the link (the part between `/d/` and `/view`)

## Using with Manual Incremental Training

Add the `--drive-ids-file` parameter to your manual incremental training commands:

```bash
# Initial training with Google Drive data
./scripts/manual_incremental.sh init 100000 --drive-ids-file drive_file_ids.json

# Continue training with Google Drive data
./scripts/manual_incremental.sh continue 2 1 100000 --drive-ids-file drive_file_ids.json
```

## Using with Automated Incremental Training

Add the `--drive-ids-file` parameter to the Python script:

```bash
python scripts/run_incremental_training.py --drive-ids-file drive_file_ids.json
```

## How It Works

The integration:

1. Automatically downloads data files from Google Drive when needed
2. Caches them locally to avoid repeated downloads
3. Uses the exact same folder structure as your existing code
4. Requires no other changes to your workflow

The file download happens transparently - if a file isn't found locally, the system will check if it has a Google Drive ID, download it, and then use it as if it was local.

## Feature Files Handling

The system handles feature data files in the following order:

1. It first attempts to load feature data directly from local files (`data/features/base_features.parquet`).
2. If not found locally, it tries to download from Google Drive using the file ID you provided in `drive_file_ids.json`.
3. If feature files cannot be found or downloaded, the system will:
   - Load raw market data (from local files or Google Drive)
   - Process this market data to extract features
   - Save the processed features for future use

This approach ensures optimal efficiency: if feature files exist (either locally or on Google Drive), they will be used directly instead of regenerating them from raw market data. This can save significant processing time, especially for complex feature calculations.

### Best Practice

For optimal performance:

1. Upload both your market data files AND pre-computed feature files to Google Drive
2. Include both in your `drive_file_ids.json` file
3. This allows the system to avoid redundant feature computation when the features are already available

Example file IDs for features:

```json
{
  "market_data/binance_BTCUSDT_5m.parquet": "1-xjib2AeyTIR4hBhQD9BDzK8fJbTcvWY",
  "market_data/binance_ETHUSDT_5m.parquet": "1dJbtH9pMy9or0h64qp7wd4UKPwLqUh95",
  "features/base_features.parquet": "1RfS1r-hWj012UkqgEi88LvnQo8U0wYrs",
  "features/base_features.meta.json": "1p1MQOiK9NOv2JCe1MG869IFqDaW_ov8T"
}
```
