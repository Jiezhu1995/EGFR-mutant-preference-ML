suppressPackageStartupMessages(library(rcdk))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: Rscript generate_cdk_regression_fingerprints.R INPUT.csv OUTPUT.csv.gz")
}

d <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
required <- c("row_index", "SMILES")
if (!all(required %in% names(d))) stop("Input requires row_index and SMILES")

to_bits <- function(fp, size) {
  value <- as.character(fp)
  if (nchar(value) != size) stop(sprintf("Expected %d bits, observed %d", size, nchar(value)))
  as.integer(strsplit(value, "", fixed = TRUE)[[1]])
}

fingerprint_one <- function(smiles) {
  tryCatch({
    mol <- parse.smiles(smiles)[[1]]
    standard <- get.fingerprint(mol, type = "standard", fp.mode = "bit")
    circular <- get.fingerprint(
      mol, type = "circular", fp.mode = "bit", depth = 6, size = 1024,
      substructure.pattern = character(), circular.type = "ECFP6", verbose = FALSE
    )
    c(to_bits(standard, 1024), to_bits(circular, 1024))
  }, error = function(e) rep.int(NA_integer_, 2048))
}

matrix <- t(vapply(d$SMILES, fingerprint_one, integer(2048)))
colnames(matrix) <- c(paste0("CDK_STANDARD_", seq_len(1024)),
                      paste0("CDK_ECFP6_", seq_len(1024)))
result <- cbind(d[required], as.data.frame(matrix, check.names = FALSE))
gz <- gzfile(args[[2]], open = "wt")
write.csv(result, gz, row.names = FALSE)
close(gz)
cat(sprintf("rows=%d valid=%d invalid=%d CDK=%s\n", nrow(d),
            sum(!is.na(matrix[, 1])), sum(is.na(matrix[, 1])), cdk.version()))

