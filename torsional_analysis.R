############################################################
# MD torsional autocorrelation – analysis script
############################################################

# 0. Setup

DATA_PATH <- "dataset/torsional_autocorrelation_final_dataset_v2.csv"

# Helper for package loading -------------------------------------------
need_pkg <- function(p, required = FALSE) {
  ok <- requireNamespace(p, quietly = TRUE)
  if (!ok && required) {
    stop(sprintf("Package '%s' is required but not installed.", p))
  }
  ok
}

# Core analysis / plotting packages (required)
need_pkg("ggplot2",  required = TRUE)
need_pkg("dplyr",    required = TRUE)
need_pkg("viridis",  required = TRUE)
need_pkg("scales",   required = TRUE)
need_pkg("broom",    required = TRUE)
need_pkg("car",      required = TRUE)

# Helpers
need_pkg("effectsize")  # for partial residual interpretation if needed

# Mixed-effects backends
has_lme4 <- need_pkg("lme4")
has_nlme <- need_pkg("nlme")

# Create plots directory if needed
if (!dir.exists("plots")) dir.create("plots")

library(ggplot2)
library(dplyr)
library(viridis)
library(scales)
library(broom)
library(car)

# 1. Load data & basic preparation 

if (!file.exists(DATA_PATH)) {
  stop(sprintf("Data file not found at: %s", DATA_PATH))
}

dat <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
cat(sprintf("\nLoaded dataset with %d rows and %d cols\n", nrow(dat), ncol(dat)))

# Helper: robust logical filter 
robust_true_filter <- function(x) {
  if (is.logical(x)) {
    return(x %in% TRUE)
  }
  if (is.numeric(x)) {
    return(!is.na(x) & x == 1)
  }
  if (is.character(x)) {
    return(tolower(x) %in% c("true", "t", "1", "yes", "y"))
  }
  rep(TRUE, length(x))
}

# 1.1 Filter to good time geometry 
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

# Fall back to unfiltered data if everything was dropped
if (nrow(dat) == 0) {
  warning("Filtering by time_ok/gap_ok removed all rows. Falling back to UNFILTERED data.")
  dat <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
}

# 1.2 Factorize categoricals
# Secondary structure
if ("seg_ss_major" %in% names(dat)) {
  dat$seg_ss_major <- factor(dat$seg_ss_major,
                             levels = c("H", "E", "L"))
}

# RMSF tertiles
if ("seg_rmsf_tertile" %in% names(dat)) {
  dat$seg_rmsf_tertile <- factor(dat$seg_rmsf_tertile,
                                 levels = c("low", "mid", "high"),
                                 ordered = TRUE)
}

# Protein ID
if ("protein_id" %in% names(dat)) {
  dat$protein_id <- factor(dat$protein_id)
} else {
  stop("dataset needs a `protein_id` column.")
}

# 1.3 Sanity checks & NA removal
needed <- c("step_w4_abs_h2", "step_global_abs_h2",
            "step_prev_abs", "abs_delta_r_mean", "seg_rmsf_mean")
missing <- setdiff(needed, names(dat))
if (length(missing) > 0) {
  stop(sprintf("These required columns are missing: %s",
               paste(missing, collapse = ", ")))
}

key_cols <- c("step_w4_abs_h2", "step_prev_abs",
              "abs_delta_r_mean", "seg_rmsf_mean")
na_mask <- apply(is.na(dat[, key_cols, drop = FALSE]), 1, any)
if (any(na_mask)) {
  cat(sprintf("Dropping %d rows with NA in key columns.\n", sum(na_mask)))
  dat <- dat[!na_mask, , drop = FALSE]
}

cat(sprintf("Final row count before modeling: %d\n", nrow(dat)))
if (nrow(dat) == 0) {
  stop("No rows left to model after NA-cleaning. Please inspect the CSV.")
}

# 1.4 Derived variables (logs for later robustness checks)
dat$log_step_w4_abs_h2     <- log(dat$step_w4_abs_h2 + 1e-6)
dat$log_step_global_abs_h2 <- log(dat$step_global_abs_h2 + 1e-6)
dat$log_abs_delta_r_mean   <- log(dat$abs_delta_r_mean + 1e-6)

# 2. Primary regression model

# Build RHS dynamically
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

# Extra-SS test for torsion disruption term
reduced_terms <- setdiff(rhs_terms, "abs_delta_r_mean")
mod_reduced <- lm(
  as.formula(paste("step_w4_abs_h2 ~", paste(reduced_terms, collapse = " + "))),
  data = dat
)

anova_primary <- anova(mod_reduced, mod_primary)
cat("\n=== Extra-SS test for abs_delta_r_mean ===\n")
print(anova_primary)

primary_coef <- summary(mod_primary)$coefficients
write.csv(primary_coef, file = "out_primary_coefficients.csv", row.names = TRUE)

# 3. Interaction models (heterogeneity)

# 3.1 Interaction with secondary structure
if ("seg_ss_major" %in% names(dat) && "seg_ss_major" %in% rhs_terms &&
    nlevels(dat$seg_ss_major) >= 2) {
  
  rhs_int   <- unique(c(rhs_terms, "abs_delta_r_mean:seg_ss_major"))
  f_int_ss  <- as.formula(paste("step_w4_abs_h2 ~", paste(rhs_int, collapse = " + ")))
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

# 3.2 Interaction with RMSF tertile 
if ("seg_rmsf_tertile" %in% names(dat)) {
  dat$seg_rmsf_tertile <- droplevels(dat$seg_rmsf_tertile)
  if (nlevels(dat$seg_rmsf_tertile) >= 2) {
    rhs_rmsf   <- unique(c(rhs_terms, "seg_rmsf_tertile",
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

# 4. Signal-regime analysis (loops, mid/high RMSF)

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

# 5. Lead–lag scan 

lag_cols <- grep("^step_w4_abs_h[0-9]+$", names(dat), value = TRUE)

lag_results <- data.frame(
  outcome   = character(),
  estimate  = numeric(),
  std_error = numeric(),
  t_value   = numeric(),
  p_value   = numeric(),
  stringsAsFactors = FALSE
)

safe_rhs <- paste(rhs_terms, collapse = " + ")

for (col in lag_cols) {
  f <- as.formula(paste(col, "~", safe_rhs))
  m <- lm(f, data = dat)
  sm <- summary(m)
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
  lag_results$p_bonf <- pmin(1, lag_results$p_value * nrow(lag_results))
  cat("\n=== LEAD–LAG RESULTS (uncorrected + Bonferroni) ===\n")
  print(lag_results)
  write.csv(lag_results, file = "out_lag_scan.csv", row.names = FALSE)
} else {
  cat("\nNo lag-like columns found for step_w4_abs_h*.\n")
}

# 6. Per-protein regressions

dat$protein_id <- droplevels(dat$protein_id)
proteins <- levels(dat$protein_id)

perprot <- data.frame(
  protein_id = character(),
  n          = integer(),
  estimate   = numeric(),
  std_error  = numeric(),
  t_value    = numeric(),
  p_value    = numeric(),
  stringsAsFactors = FALSE
)

for (p in proteins) {
  sub <- subset(dat, protein_id == p)
  if (nrow(sub) < 25) next
  
  m  <- lm(step_w4_abs_h2 ~ step_prev_abs + abs_delta_r_mean, data = sub)
  sm <- summary(m)
  if ("abs_delta_r_mean" %in% rownames(sm$coefficients)) {
    row <- sm$coefficients["abs_delta_r_mean", ]
    perprot <- rbind(
      perprot,
      data.frame(
        protein_id = p,
        n          = nrow(sub),
        estimate   = row[["Estimate"]],
        std_error  = row[["Std. Error"]],
        t_value    = row[["t value"]],
        p_value    = row[["Pr(>|t|)"]],
        stringsAsFactors = FALSE
      )
    )
  }
}

if (nrow(perprot) > 0) {
  perprot$p_bonf <- pmin(1, perprot$p_value * nrow(perprot))
  cat("\n=== PER-PROTEIN RESULTS ===\n")
  print(perprot)
  write.csv(perprot, file = "out_per_protein.csv", row.names = FALSE)
} else {
  cat("\nNo per-protein regressions were run (too few rows per protein).\n")
}

# 7. Mixed-effects model 

if (has_lme4) {
  if ("protein_id" %in% names(dat) && nlevels(dat$protein_id) >= 2) {
    cat("\n=== Mixed-effects (lme4) model ===\n")
    fe_terms <- setdiff(rhs_terms, "protein_id")
    lme4_formula <- as.formula(
      paste("step_w4_abs_h2 ~",
            paste(fe_terms, collapse = " + "),
            "+ (1 | protein_id)")
    )
    mod_lmer <- lme4::lmer(lme4_formula, data = dat, REML = TRUE)
    print(summary(mod_lmer))
    lmer_coef <- lme4::fixef(mod_lmer)
    write.csv(
      data.frame(term = names(lmer_coef),
                 estimate = as.numeric(lmer_coef)),
      file = "out_mixed_lmer_fixed_effects.csv",
      row.names = FALSE
    )
  } else {
    message("Skipping mixed model: < 2 proteins after filtering.")
  }
} else if (has_nlme) {
  if ("protein_id" %in% names(dat) && nlevels(dat$protein_id) >= 2) {
    cat("\n=== Mixed-effects (nlme) model ===\n")
    nlme_model <- nlme::lme(
      fixed  = as.formula(paste("step_w4_abs_h2 ~",
                                paste(setdiff(rhs_terms, "protein_id"),
                                      collapse = " + "))),
      random = ~ 1 | protein_id,
      data   = dat,
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

# 8. Directionality check: reverse regression

mod_reverse <- lm(
  abs_delta_r_mean ~ step_w4_abs_h2 + step_prev_abs +
    seg_rmsf_mean + seg_ss_major + protein_id,
  data = dat
)

cat("\n=== REVERSE MODEL: abs_delta_r_mean ~ future motion + covariates ===\n")
print(summary(mod_reverse))

forward <- tidy(mod_primary) |>
  dplyr::filter(term == "abs_delta_r_mean") |>
  dplyr::mutate(model = "forward")

reverse <- tidy(mod_reverse) |>
  dplyr::filter(term == "step_w4_abs_h2") |>
  dplyr::mutate(model = "reverse")

comparison <- rbind(forward, reverse)
print(comparison)

# 9. Diagnostics & time-series checks 

# VIFs for primary model (collinearity check)
cat("\n=== Variance inflation factors (primary model) ===\n")
print(car::vif(mod_primary))

# Log-model for comparison in diagnostics
mod_log <- lm(log_step_w4_abs_h2 ~ log_abs_delta_r_mean + step_prev_abs +
                seg_rmsf_mean + seg_ss_major + protein_id,
              data = dat)

# Combined 4-panel diagnostics (primary vs log)
png("plots/diag_primary_vs_log.png", width = 2400, height = 2000, res = 300)

par(
  mfrow = c(2, 2),
  mar   = c(4.5, 4.5, 3.5, 1),
  cex.axis = 1.0,
  cex.lab  = 1.1,
  cex.main = 1.1
)

# A. Residuals vs fitted – primary
plot(fitted(mod_primary), resid(mod_primary),
     xlab = "Fitted values (primary model)",
     ylab = "Residuals",
     main = "A. Residuals vs fitted (primary)")
abline(h = 0, col = "red", lty = 2)

# B. Q–Q – primary
qqnorm(resid(mod_primary),
       main = "B. Normal Q–Q (primary)")
qqline(resid(mod_primary), col = "red", lty = 2)

# C. Residuals vs fitted – log model
plot(fitted(mod_log), resid(mod_log),
     xlab = "Fitted log(future motion)",
     ylab = "Residuals",
     main = "C. Residuals vs fitted (log model)")
abline(h = 0, col = "red", lty = 2)

# D. Q–Q – log model
qqnorm(resid(mod_log),
       main = "D. Normal Q–Q (log model)")
qqline(resid(mod_log), col = "red", lty = 2)

dev.off()

# ACF of residuals (primary model)
png("plots/acf_residuals_primary.png", width = 2400, height = 1200, res = 300)
par(mar = c(4.5, 4.5, 3.5, 1),
    cex.lab = 1.2, cex.axis = 1.1, cex.main = 1.2)
acf(resid(mod_primary), main = "ACF of residuals (primary model)")
dev.off()

# 10. Figures for the manuscript
# All figures saved into ./plots/

theme_set(
  theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      panel.grid.minor = element_blank()
    )
)

# Figure 1 – Raw association: torsion change vs future motion 
p1 <- ggplot(dat,
             aes(x = abs_delta_r_mean,
                 y = step_w4_abs_h2,
                 colour = seg_rmsf_mean)) +
  geom_point(alpha = 0.35, size = 1.8) +
  scale_color_viridis_c(option = "plasma",
                        name   = "Segment flexibility (RMSF)") +
  labs(
    title = "Torsion change vs future motion (raw)",
    x = "Torsion change (|Δ torsion|)",
    y = "Future motion (next window)"
  )
ggsave("plots/fig01_torsion_vs_future_raw.png",
       plot = p1, width = 7.5, height = 5, dpi = 300)

# Figure 2 – Adjusted effect of torsion disruption (primary model)

# Reference values for covariates
ref_prev <- median(dat$step_prev_abs, na.rm = TRUE)
ref_rmsf <- median(dat$seg_rmsf_mean, na.rm = TRUE)
ref_ss   <- if ("L" %in% levels(dat$seg_ss_major)) "L" else levels(dat$seg_ss_major)[1]
ref_prot <- names(sort(table(dat$protein_id), decreasing = TRUE))[1]

tc_seq <- seq(
  quantile(dat$abs_delta_r_mean, 0.05, na.rm = TRUE),
  quantile(dat$abs_delta_r_mean, 0.95, na.rm = TRUE),
  length.out = 100
)

newdat <- data.frame(
  step_prev_abs    = ref_prev,
  abs_delta_r_mean = tc_seq,
  seg_rmsf_mean    = ref_rmsf,
  seg_ss_major     = factor(ref_ss, levels = levels(dat$seg_ss_major)),
  protein_id       = factor(ref_prot, levels = levels(dat$protein_id))
)

pred <- predict(mod_primary, newdata = newdat, se.fit = TRUE)
newdat$fit <- pred$fit
newdat$lwr <- pred$fit - 1.96 * pred$se.fit
newdat$upr <- pred$fit + 1.96 * pred$se.fit
newdat$tc_scaled <- newdat$abs_delta_r_mean * 1e5

p2 <- ggplot(newdat, aes(x = tc_scaled, y = fit)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr),
              fill = "grey80", alpha = 0.6, colour = NA) +
  geom_line(size = 1.2, colour = "#0072B2") +
  labs(
    x = expression("Torsion disruption (10"^5*" × mean |Δϕ, Δψ|, radians)"),
    y = "Predicted future segment motion",
    title = "Adjusted association between torsion disruption and future motion"
  ) +
  scale_x_continuous(breaks = pretty(newdat$tc_scaled, n = 4))

ggsave("plots/fig02_primary_effect.png",
       plot = p2, width = 7.5, height = 5, dpi = 300)

# Figure 3 – Per-protein torsion–motion slopes (forest plot)

perprot$ci_lwr <- perprot$estimate - 1.96 * perprot$std_error
perprot$ci_upr <- perprot$estimate + 1.96 * perprot$std_error

p3 <- ggplot(perprot,
             aes(x = estimate, y = reorder(protein_id, estimate))) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_pointrange(aes(xmin = ci_lwr, xmax = ci_upr),
                  size = 0.7, colour = "#333333") +
  scale_x_continuous(
    labels = label_number(scale = 1e-3, suffix = "k"),
    name = "Slope for torsion disruption\n(change in future motion per unit disruption)"
  ) +
  labs(
    y = "Protein",
    title = "Protein-specific torsion–motion slopes"
  )

ggsave("plots/fig03_per_protein_slopes.png",
       plot = p3, width = 7.5, height = 5, dpi = 300)

# Figure 4 – Global vs flexible-loop torsion effect
if (exists("mod_regime")) {
  coef_primary <- summary(mod_primary)$coefficients["abs_delta_r_mean", ]
  coef_regime  <- summary(mod_regime)$coefficients["abs_delta_r_mean", ]
  
  coef_df <- data.frame(
    model    = c("Global primary model", "Flexible-loop regime"),
    estimate = c(coef_primary["Estimate"], coef_regime["Estimate"]),
    se       = c(coef_primary["Std. Error"], coef_regime["Std. Error"])
  )
  coef_df$ci_lwr <- coef_df$estimate - 1.96 * coef_df$se
  coef_df$ci_upr <- coef_df$estimate + 1.96 * coef_df$se
  coef_df$label  <- factor(coef_df$model,
                           levels = c("Flexible-loop regime",
                                      "Global primary model"))
  
  p4 <- ggplot(coef_df, aes(x = label, y = estimate, colour = label)) +
    geom_pointrange(aes(ymin = ci_lwr, ymax = ci_upr), size = 0.9) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
    scale_y_continuous(
      labels = label_number(scale = 1e-3, suffix = "k"),
      name   = "Slope for torsion disruption"
    ) +
    scale_colour_manual(values = c("#56B4E9", "#E69F00"), guide = "none") +
    labs(
      x = "",
      title = "Torsion disruption effect: global vs flexible-loop regime"
    )
  
  ggsave("plots/fig04_global_vs_regime.png",
         plot = p4, width = 7.5, height = 5, dpi = 300)
}

# Figure 5 – Predicted torsion–motion by flexibility regime 

if (exists("mod_int_rmsf")) {
  dat$seg_rmsf_tertile <- droplevels(as.factor(dat$seg_rmsf_tertile))
  rmsf_levels <- levels(dat$seg_rmsf_tertile)
  
  tc_range <- quantile(dat$abs_delta_r_mean,
                       probs = c(0.05, 0.95),
                       na.rm = TRUE)
  tc_seq2 <- seq(tc_range[1], tc_range[2], length.out = 60)
  
  pred_grid <- expand.grid(
    abs_delta_r_mean = tc_seq2,
    seg_rmsf_tertile = rmsf_levels,
    KEEP.OUT.ATTRS   = FALSE,
    stringsAsFactors = FALSE
  )
  
  pred_grid$step_prev_abs <- ref_prev
  pred_grid$seg_rmsf_mean <- ref_rmsf
  
  pred_grid$seg_rmsf_tertile <- factor(
    pred_grid$seg_rmsf_tertile,
    levels = levels(dat$seg_rmsf_tertile)
  )
  
  example_row <- subset(dat, seg_ss_major == ref_ss & protein_id == ref_prot)
  if (nrow(example_row) == 0L) example_row <- dat[1, , drop = FALSE]
  
  pred_grid$seg_ss_major <- example_row$seg_ss_major[1]
  pred_grid$protein_id   <- example_row$protein_id[1]
  
  pg_pred <- predict(mod_int_rmsf, newdata = pred_grid, se.fit = TRUE)
  pred_grid$fit       <- pg_pred$fit
  pred_grid$tc_scaled <- pred_grid$abs_delta_r_mean * 1e5
  
  p5 <- ggplot(pred_grid,
               aes(x = tc_scaled, y = fit, colour = seg_rmsf_tertile)) +
    geom_line(size = 1.0) +
    scale_colour_brewer(palette = "Dark2", name = "RMSF tertile") +
    labs(
      x = expression("Torsion disruption (10"^5*" × mean |Δϕ, Δψ|, radians)"),
      y = "Predicted future segment motion",
      title = "Predicted torsion–motion relationship by flexibility regime"
    )
  
  ggsave("plots/fig05_rmsf_regime_lines.png",
         plot = p5, width = 7.5, height = 5, dpi = 300)
}

# Supplemental Figure – Flexibility & secondary structure boxplot 

pS1 <- ggplot(dat,
              aes(x = seg_rmsf_tertile, y = step_w4_abs_h2,
                  fill = seg_ss_major)) +
  geom_boxplot(outlier.size = 0.4) +
  scale_fill_brewer(palette = "Set2", name = "Structure type") +
  labs(
    x = "Flexibility (RMSF tertile)",
    y = "Future segment motion",
    title = "Future motion across flexibility and secondary structure"
  )

ggsave("plots/figS1_flex_ss_boxplot.png",
       plot = pS1, width = 7.5, height = 5, dpi = 300)

# Supplemental Figure – Forward vs reverse regression comparison

pS2 <- ggplot(comparison, aes(x = model, y = estimate, fill = model)) +
  geom_col(width = 0.5) +
  geom_errorbar(aes(ymin = estimate - std.error,
                    ymax = estimate + std.error),
                width = 0.2) +
  scale_fill_manual(values = c("forward" = "#4C72B0",
                               "reverse" = "#C44E52")) +
  labs(
    title = "Effect comparison: forward vs reverse regression",
    x = "Model direction",
    y = "Effect estimate",
    subtitle = "Does torsion change predict future motion, or vice versa?"
  ) +
  theme_minimal(base_size = 15)

ggsave("plots/figS2_forward_vs_reverse.png",
       plot = pS2, width = 7.5, height = 5, dpi = 300)

## Done

cat("\n=== Analysis complete. Outputs written to disk. ===\n")

