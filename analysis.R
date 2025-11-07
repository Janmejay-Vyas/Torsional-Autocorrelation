############################################################
# MD Torsional Autocorrelation — Course-Aligned Analysis
# Robust version: handles empty datasets after filtering.
############################################################

## ==== 0. Setup =====================================================
DATA_PATH <- "dataset/torsional_autocorrelation_final_dataset_v2.csv"

# Packages ------------------------------------------------------------
need_pkg <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) return(FALSE)
  TRUE
}

has_lme4 <- need_pkg("lme4")
has_nlme <- need_pkg("nlme")

## ==== 1. Load & basic prep =========================================

if (!file.exists(DATA_PATH)) {
  stop(sprintf("Data file not found at: %s", DATA_PATH))
}

dat <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

cat(sprintf("\nLoaded dataset with %d rows and %d cols\n", nrow(dat), ncol(dat)))

# Helper: robust logical filter --------------------------------------
robust_true_filter <- function(x) {
  # return logical vector marking rows that are "good"
  if (is.logical(x)) {
    return(x %in% TRUE)
  }
  if (is.numeric(x)) {
    return(!is.na(x) & x == 1)
  }
  if (is.character(x)) {
    # allow "TRUE", "True", "true", "T", "1", "yes"
    return(tolower(x) %in% c("true", "t", "1", "yes", "y"))
  }
  # if type unknown, keep all
  rep(TRUE, length(x))
}

# 1.1 Filter to good time geometry -----------------------------------
original_n <- nrow(dat)

if ("time_ok" %in% names(dat)) {
  keep_time <- robust_true_filter(dat$time_ok)
  dat <- dat[keep_time, , drop = FALSE]
  cat(sprintf("After time_ok filter: %d rows\n", nrow(dat)))
}

if ("gap_ok" %in% names(dat)) {
  keep_gap <- robust_true_filter(dat$gap_ok)
  dat <- dat[keep_gap, , drop = FALSE]
  cat(sprintf("After gap_ok filter: %d rows\n", nrow(dat)))
}

# If we lost everything, fall back to original
if (nrow(dat) == 0) {
  warning("Filtering by time_ok/gap_ok removed all rows. Falling back to UNFILTERED data.")
  dat <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
}

# 1.2 Factorize categoricals ------------------------------------------
# secondary structure
if ("seg_ss_major" %in% names(dat)) {
  dat$seg_ss_major <- factor(dat$seg_ss_major,
                             levels = c("H", "E", "L"))
}

# rmsf tertiles
if ("seg_rmsf_tertile" %in% names(dat)) {
  dat$seg_rmsf_tertile <- factor(dat$seg_rmsf_tertile,
                                 levels = c("low", "mid", "high"),
                                 ordered = TRUE)
}

# protein id
if ("protein_id" %in% names(dat)) {
  dat$protein_id <- factor(dat$protein_id)
} else {
  stop("dataset needs a `protein_id` column.")
}

# 1.3 Sanity checks for key variables --------------------------------
needed <- c("step_w4_abs_h2", "step_global_abs_h2",
            "step_prev_abs", "abs_delta_r_mean", "seg_rmsf_mean")
missing <- setdiff(needed, names(dat))
if (length(missing) > 0) {
  stop(sprintf("These required columns are missing: %s",
               paste(missing, collapse = ", ")))
}

# Drop rows with NA in key columns (common in MD exports)
key_cols <- c("step_w4_abs_h2", "step_prev_abs", "abs_delta_r_mean", "seg_rmsf_mean")
na_mask <- apply(is.na(dat[, key_cols, drop = FALSE]), 1, any)
if (any(na_mask)) {
  cat(sprintf("Dropping %d rows with NA in key columns.\n", sum(na_mask)))
  dat <- dat[!na_mask, , drop = FALSE]
}

cat(sprintf("Final row count before modeling: %d\n", nrow(dat)))
if (nrow(dat) == 0) {
  stop("No rows left to model after NA-cleaning. Please inspect the CSV.")
}

# OPTIONAL: log-transform candidates
dat$log_step_w4_abs_h2     <- log(dat$step_w4_abs_h2 + 1e-6)
dat$log_step_global_abs_h2 <- log(dat$step_global_abs_h2 + 1e-6)

## ==== 2. Primary multiple regression ================================
# We will build the RHS dynamically so we don't crash if a factor
# has only 1 observed level.

# ---- build RHS terms safely ----
rhs_terms <- c("step_prev_abs", "abs_delta_r_mean", "seg_rmsf_mean")

# seg_ss_major
if ("seg_ss_major" %in% names(dat)) {
  dat$seg_ss_major <- droplevels(dat$seg_ss_major)
  if (nlevels(dat$seg_ss_major) >= 2) {
    rhs_terms <- c(rhs_terms, "seg_ss_major")
  } else {
    message("Skipping seg_ss_major: < 2 observed levels after filtering.")
  }
}

# protein_id
if ("protein_id" %in% names(dat)) {
  dat$protein_id <- droplevels(dat$protein_id)
  if (nlevels(dat$protein_id) >= 2) {
    rhs_terms <- c(rhs_terms, "protein_id")
  } else {
    message("Skipping protein_id: < 2 observed levels after filtering.")
  }
}

primary_formula <- as.formula(
  paste("step_w4_abs_h2 ~", paste(rhs_terms, collapse = " + "))
)

mod_primary <- lm(primary_formula, data = dat)

cat("\n=== PRIMARY MODEL: step_w4_abs_h2 ~ ... ===\n")
print(summary(mod_primary))

# Test added value of abs_delta_r_mean (extra SS) ---------------------
reduced_terms <- setdiff(rhs_terms, "abs_delta_r_mean")
mod_reduced <- lm(
  as.formula(paste("step_w4_abs_h2 ~", paste(reduced_terms, collapse = " + "))),
  data = dat
)

anova_primary <- anova(mod_reduced, mod_primary)
cat("\n=== Extra-SS test for abs_delta_r_mean ===\n")
print(anova_primary)

# Save coef table
primary_coef <- summary(mod_primary)$coefficients
write.csv(primary_coef, file = "out_primary_coefficients.csv", row.names = TRUE)

## ==== 3. Interaction models (heterogeneity) =========================

### 3.1 Interaction with secondary structure --------------------------
if ("seg_ss_major" %in% names(dat) && "seg_ss_major" %in% rhs_terms &&
    nlevels(dat$seg_ss_major) >= 2) {
  
  rhs_int <- unique(c(rhs_terms, "abs_delta_r_mean:seg_ss_major"))
  f_int_ss <- as.formula(paste("step_w4_abs_h2 ~", paste(rhs_int, collapse = " + ")))
  mod_int_ss <- lm(f_int_ss, data = dat)
  
  anova_int_ss <- anova(mod_primary, mod_int_ss)
  
  cat("\n=== Interaction with seg_ss_major ===\n")
  print(summary(mod_int_ss))
  cat("\n=== Test: does torsion effect vary by SS? ===\n")
  print(anova_int_ss)
  
  write.csv(summary(mod_int_ss)$coefficients,
            file = "out_interaction_ss_coefficients.csv")
} else {
  message("Skipping SS interaction: seg_ss_major not included or < 2 levels.")
}

### 3.2 Interaction with RMSF tertile (if present) --------------------
if ("seg_rmsf_tertile" %in% names(dat)) {
  dat$seg_rmsf_tertile <- droplevels(dat$seg_rmsf_tertile)
  if (nlevels(dat$seg_rmsf_tertile) >= 2) {
    rhs_rmsf <- unique(c(rhs_terms, "seg_rmsf_tertile",
                         "abs_delta_r_mean:seg_rmsf_tertile"))
    f_int_rmsf <- as.formula(paste("step_w4_abs_h2 ~", paste(rhs_rmsf, collapse = " + ")))
    mod_int_rmsf <- lm(f_int_rmsf, data = dat)
    anova_int_rmsf <- anova(mod_primary, mod_int_rmsf)
    
    cat("\n=== Interaction with seg_rmsf_tertile ===\n")
    print(summary(mod_int_rmsf))
    cat("\n=== Test: does torsion effect vary by RMSF tertile? ===\n")
    print(anova_int_rmsf)
    
    write.csv(summary(mod_int_rmsf)$coefficients,
              file = "out_interaction_rmsf_coefficients.csv")
  } else {
    message("Skipping RMSF-tertile interaction: < 2 levels.")
  }
}

## ==== 4. Signal-regime analysis (L loops, mid/high RMSF) ============

has_signal_regime <- all(c("seg_ss_major", "seg_rmsf_tertile") %in% names(dat))

if (has_signal_regime) {
  regime <- subset(dat,
                   seg_ss_major == "L" &
                     seg_rmsf_tertile %in% c("mid", "high"))
  
  cat("\n=== SIGNAL REGIME rows:", nrow(regime), "===\n")
  
  if (nrow(regime) > 30) {
    rhs_reg <- c("step_prev_abs", "abs_delta_r_mean", "seg_rmsf_mean")
    if ("protein_id" %in% names(regime)) {
      regime$protein_id <- droplevels(regime$protein_id)
      if (nlevels(regime$protein_id) >= 2) {
        rhs_reg <- c(rhs_reg, "protein_id")
      } else {
        message("Regime: skipping protein_id (< 2 levels).")
      }
    }
    f_reg <- as.formula(paste("step_w4_abs_h2 ~", paste(rhs_reg, collapse = " + ")))
    mod_regime <- lm(f_reg, data = regime)
    cat("\n=== Model in SIGNAL REGIME ===\n")
    print(summary(mod_regime))
    write.csv(summary(mod_regime)$coefficients,
              file = "out_regime_coefficients.csv")
  } else {
    cat("Signal regime too small to fit reliably.\n")
  }
}

## ==== 5. Lead–lag scan ==============================================
# Find all columns that look like step_w4_abs_h*
lag_cols <- grep("^step_w4_abs_h[0-9]+$", names(dat), value = TRUE)

lag_results <- data.frame(
  outcome = character(),
  estimate = numeric(),
  std_error = numeric(),
  t_value = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

safe_rhs <- paste(rhs_terms, collapse = " + ")

for (col in lag_cols) {
  f <- as.formula(paste(col, "~", safe_rhs))
  m <- lm(f, data = dat)
  sm <- summary(m)
  # pull row for abs_delta_r_mean
  if ("abs_delta_r_mean" %in% rownames(sm$coefficients)) {
    row <- sm$coefficients["abs_delta_r_mean", ]
    lag_results <- rbind(
      lag_results,
      data.frame(
        outcome   = col,
        estimate  = row[["Estimate"]],
        std_error = row[["Std. Error"]],
        t_value   = row[["t value"]],
        p_value   = row[["Pr(>|t|)"]],
        stringsAsFactors = FALSE
      )
    )
  }
}

if (nrow(lag_results) > 0) {
  # Bonferroni correction within the family
  lag_results$p_bonf <- pmin(1, lag_results$p_value * nrow(lag_results))
  
  cat("\n=== LEAD–LAG RESULTS (uncorrected + Bonferroni) ===\n")
  print(lag_results)
  
  write.csv(lag_results, file = "out_lag_scan.csv", row.names = FALSE)
} else {
  cat("\nNo lag-like columns found for step_w4_abs_h*.\n")
}

## ==== 6. Per-protein regressions ====================================

dat$protein_id <- droplevels(dat$protein_id)
proteins <- levels(dat$protein_id)

perprot <- data.frame(
  protein_id = character(),
  n = integer(),
  estimate = numeric(),
  std_error = numeric(),
  t_value = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

for (p in proteins) {
  sub <- subset(dat, protein_id == p)
  if (nrow(sub) < 25) next
  
  m <- lm(step_w4_abs_h2 ~ step_prev_abs + abs_delta_r_mean,
          data = sub)
  sm <- summary(m)
  if ("abs_delta_r_mean" %in% rownames(sm$coefficients)) {
    row <- sm$coefficients["abs_delta_r_mean", ]
    perprot <- rbind(perprot,
                     data.frame(
                       protein_id = p,
                       n          = nrow(sub),
                       estimate   = row[["Estimate"]],
                       std_error  = row[["Std. Error"]],
                       t_value    = row[["t value"]],
                       p_value    = row[["Pr(>|t|)"]],
                       stringsAsFactors = FALSE
                     ))
  }
}

if (nrow(perprot) > 0) {
  # Bonferroni across proteins
  perprot$p_bonf <- pmin(1, perprot$p_value * nrow(perprot))
  
  cat("\n=== PER-PROTEIN RESULTS ===\n")
  print(perprot)
  
  write.csv(perprot, file = "out_per_protein.csv", row.names = FALSE)
} else {
  cat("\nNo per-protein regressions were run (too few rows per protein).\n")
}

## ==== 7. Optional: Mixed-effects model ==============================

if (has_lme4) {
  if ("protein_id" %in% names(dat) && nlevels(dat$protein_id) >= 2) {
    cat("\n=== Mixed-effects (lme4) model ===\n")
    # drop protein_id from fixed effects if it's in rhs_terms
    fe_terms <- setdiff(rhs_terms, "protein_id")
    lme4_formula <-
      as.formula(paste("step_w4_abs_h2 ~",
                       paste(fe_terms, collapse = " + "),
                       "+ (1 | protein_id)"))
    mod_lmer <- lme4::lmer(lme4_formula, data = dat, REML = TRUE)
    print(summary(mod_lmer))
    lmer_coef <- lme4::fixef(mod_lmer)
    write.csv(data.frame(term = names(lmer_coef),
                         estimate = as.numeric(lmer_coef)),
              file = "out_mixed_lmer_fixed_effects.csv",
              row.names = FALSE)
  } else {
    message("Skipping mixed model: < 2 proteins after filtering.")
  }
} else if (has_nlme) {
  if ("protein_id" %in% names(dat) && nlevels(dat$protein_id) >= 2) {
    cat("\n=== Mixed-effects (nlme) model ===\n")
    nlme_model <- nlme::lme(
      fixed = as.formula(paste("step_w4_abs_h2 ~",
                               paste(setdiff(rhs_terms, "protein_id"), collapse = " + "))),
      random = ~ 1 | protein_id,
      data = dat,
      method = "REML"
    )
    print(summary(nlme_model))
    write.csv(as.data.frame(nlme::fixed.effects(nlme_model)),
              file = "out_mixed_nlme_fixed_effects.csv")
  } else {
    message("Skipping mixed model: < 2 proteins after filtering.")
  }
} else {
  cat("\n(lme4/nlme not installed) — skipping mixed-effects step.\n")
}

## ==== 8. Diagnostics (basic) ========================================

png("diag_primary_resid_fitted.png", width = 700, height = 600)
plot(fitted(mod_primary), resid(mod_primary),
     xlab = "Fitted values",
     ylab = "Residuals",
     main = "Primary model: residuals vs fitted")
abline(h = 0, col = "red", lty = 2)
dev.off()

png("diag_primary_qq.png", width = 700, height = 600)
qqnorm(resid(mod_primary))
qqline(resid(mod_primary), col = "red", lty = 2)
title("QQ plot of residuals — primary model")
dev.off()

cat("\n=== Analysis complete. Outputs written to: ===\n")
cat("  - out_primary_coefficients.csv\n")
cat("  - out_interaction_ss_coefficients.csv (if run)\n")
cat("  - out_interaction_rmsf_coefficients.csv (if run)\n")
cat("  - out_regime_coefficients.csv (if regime large enough)\n")
cat("  - out_lag_scan.csv (if lag cols exist)\n")
cat("  - out_per_protein.csv (if per-protein fits ran)\n")
cat("  - diag_primary_resid_fitted.png\n")
cat("  - diag_primary_qq.png\n")
cat("  - out_mixed_lmer_fixed_effects.csv (if lme4 present & ≥2 proteins)\n")

################

library(ggplot2)

ggplot(dat, aes(x = abs_delta_r_mean, y = step_w4_abs_h2, color = seg_rmsf_mean)) +
  geom_point(alpha = 0.35, size = 1.8) +
  scale_color_viridis_c(option = "plasma") +
  labs(
    title = "Torsion change vs. Future motion (raw)",
    x = "Torsion change (|Δtorsion|)",
    y = "Future motion (later window)"
  ) +
  theme_minimal(base_size = 15)
ggsave("plot_torsion_vs_future_raw.png", width = 7.5, height = 5)


library(effectsize)

# get partial residuals ("added variable plot")
partial_resid <- residuals(mod_primary) + coef(mod_primary)["abs_delta_r_mean"] * dat$abs_delta_r_mean

ggplot(data = dat, aes(x = abs_delta_r_mean, y = partial_resid)) +
  geom_point(alpha = 0.35, size = 1.8, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red", size = 1.4) +
  labs(
    title = "Partial effect of torsion change (controlling for everything else)",
    x = "Torsion change (|Δtorsion|)",
    y = "Adjusted future motion (partial residual)"
  ) +
  theme_minimal(base_size = 15)
ggsave("plot_partial_effect_torsion.png", width = 7.5, height = 5)

library(ggplot2)
library(dplyr)
library(viridis)

dat <- read.csv("dataset/torsional_autocorrelation_final_dataset_v2.csv")

# Filter to signal regime (loops + mid/high RMSF)
regime <- dat %>%
  filter(seg_ss_major == "L",
         seg_rmsf_tertile %in% c("mid", "high"))

ggplot(regime, aes(
  x = abs_delta_r_mean,
  y = step_w4_abs_h2,
  color = seg_rmsf_mean
)) +
  geom_point(alpha = 0.45, size = 2) +
  geom_smooth(method = "lm", se = TRUE, size = 1.2, color = "black") +
  scale_color_viridis_c(option = "plasma") +
  scale_x_continuous(labels = scales::scientific) +
  labs(
    title = "Torsion Change vs Future Motion",
    x = "Magnitude of torsion disruption (|Δ torsion|)",
    y = "Future structural motion (next-time-window)",
    color = "Segment flexibility\n(RMSF)"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    panel.grid = element_blank(),            # remove grid
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave("plot_regime_torsion_future.png", width = 8, height = 5, dpi = 300)


dat$log_step_w4_abs_h2 <- log(dat$step_w4_abs_h2 + 1e-6)
dat$log_abs_delta_r_mean <- log(dat$abs_delta_r_mean + 1e-6)

mod_log <- lm(log_step_w4_abs_h2 ~ log_abs_delta_r_mean + step_prev_abs + seg_rmsf_mean + seg_ss_major + protein_id, data = dat)


qqnorm(resid(mod_log),
       main = "QQ plot of residuals",
       xlab = "Theoretical Quantiles (Normal)",
       ylab = "Sample Quantiles (Residuals)",
       pch = 19, col = "black")
qqline(resid(mod_log), col = "red", lwd = 2, lty = 2)


library(car)
vif(mod_primary)


library(ggplot2)

# Compute residuals from the primary model
resid_primary <- resid(mod_primary)

# Optional: order by time index if available
if ("window_index" %in% names(dat)) {
  dat <- dat[order(dat$window_index), ]
  resid_primary <- resid_primary[order(dat$window_index)]
}

# Remove missing residuals
resid_primary_clean <- resid_primary[!is.na(resid_primary)]

# Compute ACF on cleaned residuals
acf_data <- acf(resid_primary_clean, plot = FALSE, lag.max = 40)

# Convert to dataframe for ggplot
acf_df <- data.frame(
  Lag = acf_data$lag[-1],
  ACF = acf_data$acf[-1]
)

# Plot
library(ggplot2)
ggplot(acf_df, aes(x = Lag, y = ACF)) +
  geom_col(fill = "steelblue", width = 0.8) +
  geom_hline(yintercept = c(-0.2, 0.2), linetype = "dashed", color = "red") +
  labs(
    title = "Autocorrelation of Model Residuals",
    x = "Lag (time windows)",
    y = "Autocorrelation"
  ) +
  theme_minimal(base_size = 15)

acf(resid_primary_clean, lag.max = 10, main = "Short-term autocorrelation of residuals")


mod_reverse <- lm(
  abs_delta_r_mean ~ step_w4_abs_h2 + step_prev_abs + seg_rmsf_mean + seg_ss_major + protein_id,
  data = dat
)

summary(mod_reverse)

library(broom)

forward <- tidy(mod_primary) |> dplyr::filter(term == "abs_delta_r_mean") |> dplyr::mutate(model = "forward")
reverse <- tidy(mod_reverse) |> dplyr::filter(term == "step_w4_abs_h2") |> dplyr::mutate(model = "reverse")

comparison <- rbind(forward, reverse)
print(comparison)


library(ggplot2)

ggplot(comparison, aes(x = model, y = estimate, fill = model)) +
  geom_col(width = 0.5) +
  geom_errorbar(aes(ymin = estimate - std.error, ymax = estimate + std.error), width = 0.2) +
  scale_fill_manual(values = c("forward" = "#4C72B0", "reverse" = "#C44E52")) +
  labs(
    title = "Effect comparison: Forward vs Reverse regression",
    x = "Model direction",
    y = "Effect estimate",
    subtitle = "Does torsion change predict future motion, or vice versa?"
  ) +
  theme_minimal(base_size = 15)

