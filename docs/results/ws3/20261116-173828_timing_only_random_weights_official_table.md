Weights: random (timing only)

## Batch size 1

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res192 | reduced resolution | 144 | 4.41 | 4.44 | 0.94× | 383 |
| res208 | reduced resolution | 169 | 4.40 | 4.43 | 0.94× | 383 |
| res176 | reduced resolution | 121 | 4.40 | 4.43 | 0.94× | 383 |
| res128 | reduced resolution | 64 | 4.39 | 4.41 | 0.94× | 383 |
| res144 | reduced resolution | 81 | 4.38 | 4.42 | 0.95× | 383 |
| res160 | reduced resolution | 100 | 4.38 | 4.41 | 0.95× | 383 |
| res224 | reduced resolution | 196 | 4.37 | 4.41 | 0.95× | 383 |
| res240 | baseline | 225 | 4.14 | 4.18 | 1.00× | 384 |

## Batch size 8

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res240 | baseline | 225 | 5.95 | 5.96 | 1.00× | 679 |
| res224 | reduced resolution | 196 | 5.48 | 5.50 | 1.09× | 669 |
| res192 | reduced resolution | 144 | 5.39 | 5.41 | 1.11× | 669 |
| res176 | reduced resolution | 121 | 5.37 | 5.40 | 1.11× | 670 |
| res208 | reduced resolution | 169 | 5.35 | 5.39 | 1.11× | 669 |
| res144 | reduced resolution | 81 | 5.35 | 5.38 | 1.11× | 669 |
| res160 | reduced resolution | 100 | 5.34 | 5.37 | 1.12× | 669 |
| res128 | reduced resolution | 64 | 5.21 | 5.36 | 1.14× | 669 |

## Batch size 32

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res240 | baseline | 225 | 20.36 | 20.37 | 1.00× | 1694 |
| res224 | reduced resolution | 196 | 18.55 | 18.56 | 1.10× | 1651 |
| res208 | reduced resolution | 169 | 17.55 | 17.59 | 1.16× | 1651 |
| res192 | reduced resolution | 144 | 16.14 | 16.15 | 1.26× | 1651 |
| res176 | reduced resolution | 121 | 14.92 | 14.93 | 1.37× | 1652 |
| res160 | reduced resolution | 100 | 13.54 | 13.55 | 1.50× | 1652 |
| res144 | reduced resolution | 81 | 13.26 | 13.27 | 1.54× | 1652 |
| res128 | reduced resolution | 64 | 12.12 | 12.12 | 1.68× | 1652 |

## Batch size 128

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res240 | baseline | 225 | 96.44 | 96.47 | 1.00× | 5742 |
| res224 | reduced resolution | 196 | 88.68 | 88.72 | 1.09× | 5573 |
| res208 | reduced resolution | 169 | 79.72 | 79.76 | 1.21× | 5573 |
| res192 | reduced resolution | 144 | 71.49 | 71.52 | 1.35× | 5573 |
| res176 | reduced resolution | 121 | 63.79 | 63.82 | 1.51× | 5574 |
| res160 | reduced resolution | 100 | 57.83 | 57.85 | 1.67× | 5574 |
| res144 | reduced resolution | 81 | 53.32 | 53.35 | 1.81× | 5574 |
| res128 | reduced resolution | 64 | 48.51 | 48.53 | 1.99× | 5574 |
