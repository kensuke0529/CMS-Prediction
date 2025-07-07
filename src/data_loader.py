import requests
import pandas as pd
from pathlib import Path

def data_loader(): 
    dataset_urls = {
    2021: "https://data.cms.gov/data-api/v1/dataset/117d93f2-ce81-40fe-a4d4-8c03203b95e1/data",
    #2022: "https://data.cms.gov/data-api/v1/dataset/46bf50f8-0983-4ca2-b8d5-f2afbbf2e589/data",
    #2023: "https://data.cms.gov/data-api/v1/dataset/690ddc6c-2767-4618-b277-420ffb2bf27c/data"
    }

    limit = 1000

    # Track offsets per year
    offsets = {year: 0 for year in dataset_urls.keys()}

    # Flags to track when each year is fully fetched
    finished = {year: False for year in dataset_urls.keys()}

    all_data = []

    while not all(finished.values()):
        for year, base_url in dataset_urls.items():
            if finished[year]:
                # Skip years already finished
                continue

            params = {
                "limit": limit,
                "offset": offsets[year]
            }
            response = requests.get(base_url, params=params)

            if response.status_code != 200:
                print(f"Failed to fetch data for {year} at offset {offsets[year]}")
                finished[year] = True
                continue

            batch = response.json()
            if not batch:
                # No more data for this year
                finished[year] = True
                print(f"Finished fetching data for {year}")
                continue

            # Add year to each record
            for record in batch:
                record['year'] = year

            all_data.extend(batch)
            offsets[year] += limit
            print(
                f"Fetched {len(batch)} records for {year} at offset {offsets[year]}")

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    print(f"Total records fetched: {len(df)}")

    return df 

def main():
    base_dir = Path(__file__).resolve().parent.parent  # Assuming script is in CMS/src
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    df = data_loader()
    parquet_path = data_dir / "cms_raw.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved to {parquet_path}")

    
if __name__ == "__main__":
    main()