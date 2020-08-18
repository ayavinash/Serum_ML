library("dplyr")
library("tidyr")
library("DEP")
library("SummarizedExperiment")
library("preprocessCore")
library("EnrichmentBrowser")
library("tibble")
library("readr")
library("ggplot2")
#setwd("C:/projects/serum/working/")


## Taking all 93 samples

setwd("C:/projects/serum/new_analysis/final_IEO/ML/all_samples/results/")
data <- read.csv("C:/projects/serum/new_analysis/final_IEO/ML/all_samples/proteinGroups_filtered/proteinGroups_93_samples_0.7.txt", header = TRUE,stringsAsFactors = FALSE, sep = "\t")
expdesign <- read.csv("C:/projects/serum/new_analysis/final_IEO/ML/all_samples/93_samples.txt",header = TRUE,stringsAsFactors = FALSE, sep = "\t")



### final IEO 5 samples removed based on RT calibration  and 1 on normalization H192

setwd("C:/projects/serum/new_analysis/final_IEO/results_Davos/")
data <- read.csv("C:/projects/serum/new_analysis/final_IEO/proteinGroups_filtered/proteinGroups_expdesign_for_IEO_RT_5_samples_removed_H192_0.7.txt", header = TRUE,stringsAsFactors = FALSE, sep = "\t")
expdesign <- read.csv("C:/projects/serum/new_analysis/final_IEO/expdesign_for_IEO_RT_5_samples_removed_H192.txt",header = TRUE,stringsAsFactors = FALSE, sep = "\t")
#expdesign <- read.csv("C:/projects/serum/new_analysis/final_IEO/expdesign_for_IEO_RT_5_samples_removed_H192_rep.txt",header = TRUE,stringsAsFactors = FALSE, sep = "\t")


data_se <- import_MaxQuant(data, expdesign, filter = c("Reverse",
                                                      "Potential.contaminant"), intensities = "LFQ", names = "Gene.names",
                          ids = "Protein.IDs", delim = ";")


data_norm <- normalize_vsn(data_se)
set.seed(21)
data_imp <- impute(data_norm, fun = "man", shift = 1.8, scale = 0.3)
pdf("missing_values.pdf",  width=15, height=7)
plot_missval(data_norm)
dev.off()
pdf("missing_freq.pdf",  width=15, height=7)
plot_frequency(data_se)
dev.off()

pdf("histogram_missing_values.pdf",  width=7, height=7)
plot_detect(data_se)
dev.off()
pdf("imputation.pdf",  width=7, height=7)
plot_imputation(data_se,data_imp)
dev.off()
data_diff <- test_diff(data_imp, type = "man",test =c("Tumor_vs_Healthy"))
dep <- add_rejections(data_diff, alpha = 0.05, lfc =1)
pdf("volcano.pdf",  width=7, height=7)
plot_volcano(dep, contrast = "Tumor_vs_Healthy", label_size = 5, add_names = TRUE)
dev.off()
pdf("volcano_no_labels.pdf",  width=7, height=7)
plot_volcano(dep, contrast = "Tumor_vs_Healthy", label_size = 5, add_names = FALSE)
dev.off()
pdf("volcano_adjusted_y_axis.pdf",  width=7, height=7)
plot_volcano(dep, contrast = "Tumor_vs_Healthy", label_size = 5, add_names = TRUE,adjusted=TRUE)
dev.off()


results<-get_results(dep)
write.table(results, "Diff_genes.txt", sep="\t", quote=FALSE, row.names=FALSE)
length(results$significant[results$significant])
pdf("protein_counts.pdf",  width=15, height=7)
plot_numbers(data_se)
dev.off()
pdf("normalization.pdf",  width=15, height=15)
plot_normalization(data_norm)
dev.off()
pdf("pca.pdf",  width=7, height=7)
plot_pca(dep, x = 1, y = 2,indicate="condition",label=FALSE,n=length(results$significant[results$significant]))
dev.off()
pdf("corr_deg.pdf",  width=15, height=15)
plot_cor(dep, significant = TRUE, lower = 0, upper = 1, pal = "Reds",font_size=10)
dev.off()
pdf("corr_all.pdf",  width=15, height=15)
plot_cor(dep, significant = FALSE, lower = 0, upper = 1, pal = "Reds",font_size=10)
dev.off()

pdf("heatmap.pdf",  width=15, height=7)
plot_heatmap(dep, type = "centered", kmeans = TRUE, 
             k = 6, col_limit = 4, show_row_names = TRUE,
             indicate = c("condition"))
dev.off()
#for (protein in c(results$name[results$significant])) {
#    print(protein)
#    pdf(paste(protein,".pdf"),  width=15, height=7)
#    print(plot_single(dep,protein,type="centered"))
#    dev.off()
                }


##### changes for DAVOS

my_plot_volcano <- function(dep, contrast, label_size = 3,
                         add_names = TRUE, adjusted = FALSE, plot = TRUE) {
  # Show error if inputs are not the required classes
  if(is.integer(label_size)) label_size <- as.numeric(label_size)
  assertthat::assert_that(inherits(dep, "SummarizedExperiment"),
                          is.character(contrast),
                          length(contrast) == 1,
                          is.numeric(label_size),
                          length(label_size) == 1,
                          is.logical(add_names),
                          length(add_names) == 1,
                          is.logical(adjusted),
                          length(adjusted) == 1,
                          is.logical(plot),
                          length(plot) == 1)
  
  row_data <- rowData(dep, use.names = FALSE)
  
  # Show error if inputs do not contain required columns
  if(any(!c("name", "ID") %in% colnames(row_data))) {
    stop(paste0("'name' and/or 'ID' columns are not present in '",
                deparse(substitute(dep)),
                "'.\nRun make_unique() to obtain required columns."),
         call. = FALSE)
  }
  if(length(grep("_p.adj|_diff", colnames(row_data))) < 1) {
    stop(paste0("'[contrast]_diff' and '[contrast]_p.adj' columns are not present in '",
                deparse(substitute(dep)),
                "'.\nRun test_diff() to obtain the required columns."),
         call. = FALSE)
  }
  if(length(grep("_significant", colnames(row_data))) < 1) {
    stop(paste0("'[contrast]_significant' columns are not present in '",
                deparse(substitute(dep)),
                "'.\nRun add_rejections() to obtain the required columns."),
         call. = FALSE)
  }
  
  # Show error if an unvalid contrast is given
  if (length(grep(paste(contrast, "_diff", sep = ""),
                  colnames(row_data))) == 0) {
    valid_cntrsts <- row_data %>%
      data.frame() %>%
      select(ends_with("_diff")) %>%
      colnames(.) %>%
      gsub("_diff", "", .)
    valid_cntrsts_msg <- paste0("Valid contrasts are: '",
                                paste0(valid_cntrsts, collapse = "', '"),
                                "'")
    stop("Not a valid contrast, please run `plot_volcano()` with a valid contrast as argument\n",
         valid_cntrsts_msg,
         call. = FALSE)
  }
  
  # Generate a data.frame containing all info for the volcano plot
  diff <- grep(paste(contrast, "_diff", sep = ""),
               colnames(row_data))
  if(adjusted) {
    p_values <- grep(paste(contrast, "_p.adj", sep = ""),
                     colnames(row_data))
  } else {
    p_values <- grep(paste(contrast, "_p.val", sep = ""),
                     colnames(row_data))
  }
  signif <- grep(paste(contrast, "_significant", sep = ""),
                 colnames(row_data))
  df <- data.frame(x = row_data[, diff],
                   y = -log10(row_data[, p_values]),
                   significant = row_data[, signif],
                   name = row_data$name) %>%
    filter(!is.na(significant)) %>%
    arrange(significant)
  
  name1 <- gsub("_vs_.*", "", contrast)
  name2 <- gsub(".*_vs_", "", contrast)
  
  # Plot volcano with or without labels
  p <- ggplot(df, aes(x, y)) +
    geom_vline(xintercept = 0) +
    geom_point(aes(col = significant)) +
    geom_text(data = data.frame(), aes(x = c(Inf, -Inf),
                                       y = c(-Inf, -Inf),
                                       hjust = c(1, 0),
                                       vjust = c(-1, -1),
                                       label = c(name1, name2),
                                       size = 5,
                                       fontface = "bold")) +
    labs(title = contrast,
         x = expression(log[2]~"Fold change")) +
    theme_DEP1() +
    theme(legend.position = "none") +
    scale_color_manual(values = c("TRUE" = "red", "FALSE" = "grey"))
  if (add_names) {
    p <- p + ggrepel::geom_text_repel(data = filter(df, significant),
                                      aes(label = name),
                                      size = label_size,
                                      box.padding = unit(0.1, 'lines'),
                                      point.padding = unit(0.1, 'lines'),
                                      segment.size = 0.5)
  }
  if(adjusted) {
    p <- p + labs(y = expression(-log[10]~"Adjusted p-value"))
  } else {
    p <- p + labs(y = expression(-log[10]~"P-value"))
  }
  if(plot) {
    return(p)
  } else {
    df <- df %>%
      select(name, x, y, significant) %>%
      arrange(desc(x))
    colnames(df)[c(1,2,3)] <- c("protein", "log2_fold_change", "p_value_-log10")
    if(adjusted) {
      colnames(df)[3] <- "adjusted_p_value_-log10"
    }
    return(df)
  }
}

pdf("volcano_red.pdf",  width=7, height=7)
my_plot_volcano(dep, contrast = "Tumor_vs_Healthy", label_size = 5, add_names = TRUE)
dev.off()
pdf("volcano_red_no_labels.pdf",  width=7, height=7)
my_plot_volcano(dep, contrast = "Tumor_vs_Healthy", label_size = 5, add_names = FALSE)
dev.off()





my_plot_single <- function(dep, proteins, type = c("contrast", "centered"), plot = TRUE) {
  # Show error if inputs are not the required classes
  assertthat::assert_that(inherits(dep, "SummarizedExperiment"),
    is.character(proteins),
    is.character(type),
    is.logical(plot),
    length(plot) == 1)

  # Show error if inputs do not contain required columns
  type <- match.arg(type)

  row_data <- rowData(dep, use.names = FALSE)

  if(any(!c("label", "condition", "replicate") %in% colnames(colData(dep)))) {
    stop("'label', 'condition' and/or 'replicate' columns are not present in '",
         deparse(substitute(dep)),
         "'\nRun make_se() or make_se_parse() to obtain the required columns",
         call. = FALSE)
  }
  if(length(grep("_p.adj|_diff", colnames(row_data))) < 1) {
    stop("'[contrast]_diff' and '[contrast]_p.adj' columns are not present in '",
         deparse(substitute(dep)),
         "'\nRun test_diff() to obtain the required columns",
         call. = FALSE)
  }
  if(!"name" %in% colnames(row_data)) {
    stop("'name' column not present in '",
         deparse(substitute(dep)),
         "'\nRun make_se() or make_se_parse() to obtain the required columns",
         call. = FALSE)
  }

  # Show error if an unvalid protein name is given
  if(all(!proteins %in% row_data$name)) {
    if(length(proteins) == 1) {
      rows <- grep(substr(proteins, 1, nchar(proteins) - 1),row_data$name)
      possibilities <- row_data$name[rows]
    } else {
      rows <- lapply(proteins, function(x)
        grep(substr(x, 1, nchar(x) - 1),row_data$name))
      possibilities <- row_data$name[unlist(rows)]
    }

    if(length(possibilities) > 0) {
      possibilities_msg <- paste0(
        "Do you mean: '",
        paste0(possibilities, collapse = "', '"),
        "'")
    } else {
      possibilities_msg <- NULL
    }
    stop("please run `plot_single()` with a valid protein names in the 'proteins' argument\n",
         possibilities_msg,
         call. = FALSE)
  }
  if(any(!proteins %in% row_data$name)) {
    proteins <- proteins[proteins %in% row_data$name]
    warning("Only used the following protein(s): '",
            paste0(proteins, collapse = "', '"),
            "'")
  }

  # Single protein
  subset <- dep[proteins]

  # Plot either the centered log-intensity values
  # per condition ('centered') or the average fold change of conditions
  # versus the control condition ('contrast') for a single protein
  if(type == "centered") {
    # Obtain protein-centered fold change values
    means <- rowMeans(assay(subset), na.rm = TRUE)
    df_reps <- data.frame(assay(subset) - means) %>%
      rownames_to_column() %>%
      gather(ID, val, -rowname) %>%
      left_join(., data.frame(colData(subset)), by = "ID")
    df_reps$replicate <- as.factor(df_reps$replicate)
    df <- df_reps %>%
      group_by(condition, rowname) %>%
      summarize(mean = mean(val, na.rm = TRUE),
        sd = sd(val, na.rm = TRUE),
        n = n()) %>%
      mutate(error = qnorm(0.975) * sd / sqrt(n),
             CI.L = mean - error,
             CI.R = mean + error) %>%
      as.data.frame()
    df$rowname <- parse_factor(df$rowname, levels = proteins)

    # Plot the centered intensity values for the replicates and the mean
    p <- ggplot(df, aes(condition, mean)) +
      geom_hline(yintercept = 0) +
      geom_col(colour = "black", fill = "grey") +
      geom_point(data = df_reps, aes(condition, val, col = replicate), shape = 18, size = 5, show.legend=FALSE,position = position_dodge(width=0.3)) +
      geom_errorbar(aes(ymin = CI.L, ymax = CI.R), width = 0.3) +
      labs(x = "Condition",
           y = expression(log[2]~"Centered intensity"~"(\u00B195% CI)"),col=NULL) +facet_wrap(~rowname)+theme_DEP1()}
  if(type == "contrast") {
    # Select values for a single protein
    df <- rowData(subset, use.names = FALSE) %>%
      data.frame() %>%
      select(name,
             ends_with("_diff"),
             ends_with("_CI.L"),
             ends_with("_CI.R")) %>%
      gather(var, val, -name) %>%
      mutate(contrast = gsub("_diff|_CI.L|_CI.R", "", var),
             var = gsub(".*_", "", var)) %>%
      spread(var, val)
    df$name <- parse_factor(df$name, levels = proteins)
    suffix <- get_suffix(df$contrast)
    if(length(suffix)) {df$contrast <- delete_suffix(df$contrast)}
    # Plot the average fold change of conditions versus the control condition
    p <- ggplot(df, aes(contrast, diff)) +
      geom_hline(yintercept = 0) +
      geom_col(colour = "black", fill = "grey") +
      geom_errorbar(aes(ymin = CI.L, ymax = CI.R), width = 0.3) +
      labs(x = suffix,
           y = expression(log[2]~"Fold change"~"(\u00B195% CI)")) +
      facet_wrap(~name) +
      theme_DEP2()
  }
  if(plot) {
    return(p)
  } else {
    if(type == "centered") {
      df <- df %>%
        select(rowname, condition, mean, CI.L, CI.R)
      colnames(df) <- c("protein", "condition",
        "log2_intensity", "CI.L", "CI.R")
    }
    if(type == "contrast") {
      df <- df %>%
        select(name, contrast, diff, CI.L, CI.R) %>%
        mutate(contrast = paste0(contrast, suffix))
      colnames(df) <- c("protein", "contrast",
        "log2_fold_change", "CI.L", "CI.R")
    }
    return(df)
  }
}


for (protein in c(results$name[results$significant])) {
    print(protein)
    pdf(paste(protein,".pdf"),  width=7, height=7)
    print(my_plot_single(dep,protein,type="centered"))
    dev.off()
                }

