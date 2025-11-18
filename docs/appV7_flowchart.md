# AppV7 Process Flow

```mermaid
flowchart TB
    %% Input
    A["CAD File<br/><small>.dxf/.dwg/.pdf/.step</small>"] --> B & C & D & E1

    %% Primary Extraction
    B["Keyword Finder<br/><small>KeywordDetector.py</small>"]
    C["GEO Dump<br/><small>geo_dump.py</small>"]
    D["PaddleOCR<br/><small>paddle_dims_extractor.py</small>"]
    E1["CAD Feature Extractor<br/><small>cad_feature_extractor.py</small>"]

    %% Secondary Processing
    B --> E & G
    C --> F & E & G & H1
    D --> E
    E1 --> E & G & H1

    E["Direct Costs<br/><small>DirectCostHelper.py</small>"]
    F["GEO Page<br/><small>geo_extractor.py</small>"]
    G["Planner Picker<br/><small>process_planner.py<br/>pick_family_and_hints()</small>"]
    H1["Hole Ops<br/><small>hole_ops.py</small>"]

    %% Part Analysis
    E --> H
    H["Part Size<br/><small>DirectCostHelper.py</small>"]

    %% Planning
    G --> I
    I["Process Planner<br/><small>process_planner.py</small>"]
    I --> K & L

    %% Operations
    K["Machine Ops<br/><small>time_estimator.py</small>"]
    L["Labor Ops<br/><small>process_planner.py<br/>labor_minutes()</small>"]

    %% Material Flow
    H --> J
    J["Material Mapper<br/><small>MaterialMapper.py</small>"]
    J --> n1 & n2

    %% McMaster Path
    n1["McMaster Catalog<br/><small>mcmaster_helpers.py</small>"]
    n1 --> n3
    n3["McMaster API<br/><small>mcmaster_api.py</small>"]

    %% Material Data
    n2["Material Density<br/><small>material_density.py<br/>resources/materials.json</small>"]
    n2 --> n4

    %% Scrap Calculations
    n4["Scrap Maths<br/><small>scrap_pricing.py</small>"]
    n4 --> n5
    n5["Wieland Scraper<br/><small>wieland_scraper.py</small>"]
    n5 --> n12
    n12["Scrap Value<br/><small>(Credit)</small>"]

    %% Machine Cost Path
    K --> n6
    n6["Task Processing<br/><small>time_estimator.py</small>"]
    n6 --> n8 & n15 & R1

    %% Rate Infrastructure
    R1["Rate Buckets<br/><small>rate_buckets.py<br/>rates.py</small>"]
    R1 --> n9 & n10

    %% Speeds and Feeds
    n15["Speeds & Feeds<br/><small>speeds_feeds_selector.py<br/>speeds_feeds_merged.csv</small>"]
    n15 --> n8

    %% Operation Tables
    n8["Hole/Tap/Cut Tables<br/><small>hole_ops.py<br/>resources/*.json</small>"]
    H1 --> n8
    n8 --> n9 & n4

    %% Machine Cost
    n9["Sum Machine Cost<br/><small>rate_buckets.py</small>"]

    %% Labor Path
    L --> n7
    n7["Hours Table<br/><small>Master_Variables.csv</small>"]
    n7 --> n10

    %% Labor Cost
    n10["Sum Labor Cost<br/><small>rate_buckets.py</small>"]

    %% Material Cost Aggregation
    n12 --> n13
    n3 --> n13
    n16["Shipping<br/><small>12.5% on stock</small>"] --> n13
    n17["Tax<br/><small>7% on stock</small>"] --> n13
    n13["Sum Material Cost<br/><small>QuoteDataHelper.py</small>"]

    %% Cost Summary
    n9 --> n11
    n10 --> n11
    n13 --> n11

    %% Overrides & Final
    n11["Cost Summary<br/><small>QuoteDataHelper.py<br/>CostSummary class</small>"]

    OV["LLM Overrides<br/><small>llm_overrides.py<br/>llm_suggest.py</small>"]
    OV --> n11

    n11 --> n18

    %% Quantity & Margin
    QTY["Quantity Amortization<br/><small>Setup ÷ Qty</small>"] --> n18
    MRG["Margin<br/><small>Multiplicative</small>"] --> n18

    n18["Final Price<br/><small>QuoteDataHelper.py</small>"]

    %% Output
    n18 --> OUT
    OUT["Quote Output<br/><small>AppV7.py GUI</small>"]

    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b
    classDef extraction fill:#f3e5f5,stroke:#4a148c
    classDef processing fill:#fff3e0,stroke:#e65100
    classDef data fill:#e8f5e9,stroke:#1b5e20
    classDef cost fill:#fce4ec,stroke:#880e4f
    classDef output fill:#f5f5f5,stroke:#212121

    class A input
    class B,C,D,E1,H1 extraction
    class E,F,G,H,I,J,K,L processing
    class n1,n2,n3,n5,n7,n8,n15,R1 data
    class n4,n6,n9,n10,n12,n13,n11,OV,QTY,MRG cost
    class n18,OUT output
```

## Component Details

### Input Processing
| Component | File | Description |
|-----------|------|-------------|
| CAD File | - | Input files (.dxf, .dwg, .pdf, .step) |
| Keyword Finder | `cad_quoter/pricing/KeywordDetector.py` | Detects process keywords from CAD |
| GEO Dump | `cad_quoter/geo_dump.py` | Extracts geometric data |
| PaddleOCR | `tools/paddle_dims_extractor.py` | OCR for dimension extraction |
| CAD Feature Extractor | `cad_quoter/cad_feature_extractor.py` | Comprehensive feature analysis |

### Planning & Operations
| Component | File | Description |
|-----------|------|-------------|
| Direct Costs | `cad_quoter/pricing/DirectCostHelper.py` | Direct cost calculations |
| GEO Page | `cad_quoter/geo_extractor.py` | GEO page extraction |
| Planner Picker | `cad_quoter/planning/process_planner.py` | `pick_family_and_hints()` function |
| Process Planner | `cad_quoter/planning/process_planner.py` | Main planning logic |
| Machine Ops | `cad_quoter/pricing/time_estimator.py` | Machine time estimation |
| Labor Ops | `cad_quoter/planning/process_planner.py` | `labor_minutes()` functions |
| Hole Ops | `tools/hole_ops.py` | Hole/tap/cut operations |

### Material & Pricing
| Component | File | Description |
|-----------|------|-------------|
| Material Mapper | `cad_quoter/pricing/MaterialMapper.py` | Maps materials to sources |
| McMaster Catalog | `cad_quoter/pricing/mcmaster_helpers.py` | McMaster catalog lookup |
| McMaster API | `cad_quoter/mcmaster_api.py` | API interface |
| Material Density | `cad_quoter/material_density.py` | Density calculations |
| Scrap Pricing | `cad_quoter/pricing/scrap_pricing.py` | Scrap value calculations |
| Wieland Scraper | `cad_quoter/pricing/wieland_scraper.py` | Scrap price scraping |

### Cost Calculation
| Component | File | Description |
|-----------|------|-------------|
| Rate Buckets | `cad_quoter/pricing/rate_buckets.py` | Machine/labor rates |
| Rates | `cad_quoter/pricing/rates.py` | Rate definitions |
| Speeds & Feeds | `cad_quoter/pricing/speeds_feeds_selector.py` | Machining parameters |
| Cost Summary | `cad_quoter/pricing/QuoteDataHelper.py` | `CostSummary` class |

### Overrides & Output
| Component | File | Description |
|-----------|------|-------------|
| LLM Overrides | `cad_quoter/llm_overrides.py` | AI-driven adjustments |
| LLM Suggest | `cad_quoter/llm_suggest.py` | AI suggestions |
| Final Price | `cad_quoter/pricing/QuoteDataHelper.py` | Final quote calculation |
| GUI Output | `AppV7.py` | Tkinter GUI display |

## Key Data Files
- `cad_quoter/resources/Master_Variables.csv` - Rates and variables
- `cad_quoter/pricing/resources/speeds_feeds_merged.csv` - Machining data
- `cad_quoter/resources/materials.json` - Material definitions
- Various `.json` configs for drilling, amortization, etc.

## Business Logic Notes
- **Tax**: 7% applied to McMaster stock
- **Shipping**: 12.5% applied to McMaster stock
- **Scrap**: Calculated as credit (subtracted from cost)
- **Margin**: Applied multiplicatively to total
- **Quantity**: Setup costs amortized by quantity
