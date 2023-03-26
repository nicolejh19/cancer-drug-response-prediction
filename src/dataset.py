import numpy as np
import pandas as pd

import csv
import logging
import os
import torch

from functools import cached_property
from itertools import product


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset

DRUG_SMILES_PATH = "../data/raw/drug_smiles.csv"
DATA_BASE_DIR = "/home/nicole/cancer_data_analysis/variant_annotations/data/"

RANDOM_STATE = 31

GENES_324 = pd.read_csv("../data/gene2ind.txt", header=None)[
    0
].tolist()

cell_line_auc_df = pd.read_csv("../data/cell_drug_auc_final_1111.csv")
cell_line_auc_df["depmap_id"] = cell_line_auc_df["ARXSPAN_ID"].astype("string")
cell_line_auc_df.drop("ARXSPAN_ID", axis=1, inplace=True)
cell_line_auc_df.set_index(["depmap_id"], inplace=True)
CELL_LINE_DRUGS_ALL = cell_line_auc_df.columns.tolist()


fname_drugs_cat = "../data/druid-druglist.csv"
df_drugs_cat = pd.read_csv(fname_drugs_cat)
list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])]["drug_name"].tolist()

CELL_LINE_DRUGS_CAT_1 = list(
    set(CELL_LINE_DRUGS_ALL).intersection(set(list_drugs_cat1))
)

class CellLineDataset(Dataset):
    """
    Base class for datasets that hold cell line information
    """

    base_dir = DATA_BASE_DIR
    entity_identifier_name = None

    def __str__(self):
        dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
        return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
        #Entities - {len(dataset_df[self.entity_identifier_name].unique())}
        #Drugs - {len(dataset_df.drug_name.unique())}
        #Pairs - {len(self)}
        """

    pass


class AnnotatedCellLineDataset(CellLineDataset):
    """
    Cell line data with annotation features from annovar
    """

    entity_identifier_name = "depmap_id"

    def __init__(self, is_train=True, only_cat_one_drugs=True):
        """
        Parameters
        ----------
        is_train : bool
            Returns items from the train or test split (defaults to True)
        only_cat_one_drugs : bool
            Filters for items corresponding to category 1 drugs (defaults to True)

        """
        ccle_df = pd.read_csv(
            "../data/processed/CCLE_hg38_DepMap21Q3_annot_subset_cols_numeric_imputed_filtered.csv"
        )
        ccle_df.set_index(["depmap_id", "input"], inplace=True)
        numeric_columns = list(
            column
            for column in ccle_df.columns
            if pd.api.types.is_numeric_dtype(ccle_df[column])
        )
        self.ccle_df = ccle_df[numeric_columns].copy()

        df_auc = pd.read_csv("../data/cell_drug_auc_final_1111.csv")
        df_auc["depmap_id"] = df_auc["ARXSPAN_ID"].astype("string")
        df_auc.drop("ARXSPAN_ID", axis=1, inplace=True)
        df_auc.set_index(["depmap_id"], inplace=True)

        # Filter for category one drug, if needed
        self.only_cat_one_drugs = only_cat_one_drugs
        if self.only_cat_one_drugs:
            fname_drugs_cat = "../data/druid-druglist.csv"
            df_drugs_cat = pd.read_csv(fname_drugs_cat)
            list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])][
                "drug_name"
            ].tolist()
            filtered_drugs = [col for col in df_auc.columns if col in list_drugs_cat1]
            df_auc = df_auc[filtered_drugs]

        self.is_train = is_train
        if self.is_train:
            required_cell_line_ids = pd.read_csv(
                "../data/train_celllines_v1_1111", header=None
            )[0].values
        else:
            required_cell_line_ids = pd.read_csv(
                "../data/test_celllines_v1_1111", header=None
            )[0].values

        y_df = df_auc[df_auc.index.isin(required_cell_line_ids)].copy()

        # The below filter is to remove those cellines for which there are no
        # annotation features available (likely due to the absence of point
        # mutations in such cases)
        #
        # TODO: Check how to represent those cases that do not have any point
        # mutations
        y_df = y_df[y_df.index.isin(ccle_df.index.get_level_values(0))].copy()

        y_df = y_df.reset_index().melt(
            id_vars=[
                "depmap_id",
            ],
            var_name="drug_name",
            value_name="auc",
        )
        self.y_df = y_df[~(y_df.auc < 0)]

    def __len__(self):
        return len(self.y_df)

    def __getitem__(self, idx):
        record = self.y_df.iloc[idx]

        return {
            "depmap_id": record["depmap_id"],
            "drug_name": record["drug_name"],
            "auc": record["auc"],
        }

    @cached_property
    def annotations(self):
        return self.ccle_df

    @cached_property
    def mutations(self):
        df_reprn_mut = pd.read_csv(
            "../data/processed/ccle_raw_mutation.csv",
        )
        df_reprn_mut.set_index(["depmap_id"], inplace=True)
        df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
        return df_reprn_mut

    @cached_property
    def cnv(self):
        cnv_df = pd.read_csv(
            "../data/processed/ccle_cnv_binary.csv",
        )
        cnv_df.set_index(["depmap_id"], inplace=True)
        cnv_df = cnv_df.reindex(columns=GENES_324)
        return cnv_df

    @cached_property
    def gene_exp(self):
        gene_exp_df = pd.read_csv("../data/processed/ccle_gene_expression.csv")
        gene_exp_df.rename(columns={"cell_id": "depmap_id"}, inplace=True)
        gene_exp_df.set_index("depmap_id", drop=True, inplace=True)
        return gene_exp_df[GENES_324]


class PdxDataset(Dataset):
    """
    Base class for datasets that hold PDX information
    """

    base_dir = DATA_BASE_DIR

    def __str__(self):
        dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
        return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
        #Entities - {len(dataset_df[self.entity_identifier_name].unique())}
        #Drugs - {len(dataset_df.drug_name.unique())}
        #Pairs - {len(self)}
        #Response (0 to 1) - {dataset_df.response.value_counts()[0]} to {dataset_df.response.value_counts()[1]}
        """

    pass


class AnnotatedPdxDataset(PdxDataset):
    """
    PDX data, used only for testing
    """

    entity_identifier_name = "Sample"

    def __init__(self, apply_train_test_filter=True, is_train=False):
        self.is_train = is_train

        pdx_response = pd.read_csv("../data/pdx_drug_response.csv")

        # TODO: Find smile strings for the missing drugs and update the csv
        drug_smiles = csv.reader(open(DRUG_SMILES_PATH))

        drug_names = [item[0] for item in drug_smiles]
        pdx_response = pdx_response[pdx_response["Treatment"].isin(drug_names)]

        pdx_mutation = pd.read_csv(
            "../data/pdx_mutations_samples_with_drug.csv"
        )
        pdx_mutation.set_index("Sample", inplace=True)
        self.pdx_mutation_filtered = pdx_mutation.reindex(GENES_324, axis="columns")

        self.pdx_response = pdx_response[["Sample", "Treatment", "response"]].copy()
        self.pdx_response.rename(columns={"Treatment": "drug_name"}, inplace=True)

        if apply_train_test_filter:
            uniq_sample_ids = self.pdx_response.Sample.unique()
            train_ids, test_ids, _, _ = train_test_split(
                uniq_sample_ids,
                np.arange(len(uniq_sample_ids)),
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
            filter_ids = train_ids if is_train else test_ids
            self.pdx_response = self.pdx_response[
                self.pdx_response.Sample.isin(filter_ids)
            ].copy()

    def __len__(self):
        return len(self.pdx_response)

    def __getitem__(self, idx):
        record = self.pdx_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
        }

    @cached_property
    def annotations(self):
        pdx_anno_features = pd.read_csv(
            "../data/processed/pdx_anno_features_imputed.csv"
        )
        pdx_anno_features.set_index(
            [self.entity_identifier_name, "input"], inplace=True
        )
        return pdx_anno_features

    @cached_property
    def mutations(self):
        return self.pdx_mutation_filtered

    @cached_property
    def cnv(self):
        cnv_df = pd.read_csv(
            "../data/processed/pdx_cnv.csv",
        )
        cnv_df.set_index(["Sample"], inplace=True)
        cnv_df = cnv_df.reindex(columns=GENES_324)
        # CNV represents no change/loss/amplification with 0/-1/1. For the purpose
        # of distance computation, it is sufficient if they are only
        # indicator variables (0/1) and the below does the same
        cnv_df.replace(to_replace=-1, value=1, inplace=True)
        return cnv_df


class CategoricalAnnotatedCellLineDataset(CellLineDataset):
    """
    Cell line data with categorical annotation features from annovar
    """

    entity_identifier_name = "depmap_id"

    def __init__(
        self,
        is_train=True,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
    ):
        """
        Parameters
        ----------
        is_train : bool
            Returns items from the train or test split (defaults to True)
        only_cat_one_drugs : bool
            Filters for items corresponding to category 1 drugs (defaults to True)

        """
        self.is_train = is_train
        self.df_reprn_mut = pd.read_csv(
            "../data/processed/ccle_anno_features.csv",
        )
        self.df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in self.df_reprn_mut.columns:
            mask = mask & (self.df_reprn_mut[col] == 0)

        self.df_reprn_mut = self.df_reprn_mut[~mask].copy()

        df_auc = pd.read_csv("../data/cell_drug_auc_final_1111.csv")
        df_auc["depmap_id"] = df_auc["ARXSPAN_ID"].astype("string")
        df_auc.drop("ARXSPAN_ID", axis=1, inplace=True)
        df_auc.set_index(["depmap_id"], inplace=True)

        # Filter for category one drug, if needed
        self.only_cat_one_drugs = only_cat_one_drugs
        if self.only_cat_one_drugs:
            fname_drugs_cat = "../data/druid-druglist.csv"
            df_drugs_cat = pd.read_csv(fname_drugs_cat)
            list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])][
                "drug_name"
            ].tolist()
            # The below two drugs are effective in a lot of cell-lines which skews the
            # predictions considerably. Hence, we remove these two drugs from training
            list_drugs_cat1.remove("PAZOPANIB")
            list_drugs_cat1.remove("IMATINIB")
            filtered_drugs = [col for col in df_auc.columns if col in list_drugs_cat1]
            df_auc = df_auc[filtered_drugs]

        train_cell_lines_ids = pd.read_csv(
            "../data/train_celllines_v1_1111", header=None
        )[0].values

        test_cell_lines_ids = pd.read_csv(
            "../data/test_celllines_v1_1111", header=None
        )[0].values

        if is_train is not None:
            if self.is_train:
                required_cell_line_ids = train_cell_lines_ids
            else:
                required_cell_line_ids = test_cell_lines_ids
        else:
            required_cell_line_ids = np.concatenate(
                [train_cell_lines_ids, test_cell_lines_ids]
            )

        if scale_y:
            y_train_df = df_auc[df_auc.index.isin(train_cell_lines_ids)].copy()
            y_train = y_train_df.to_numpy()
            y_test_df = df_auc[df_auc.index.isin(test_cell_lines_ids)].copy()
            y_test = y_test_df.to_numpy()
            y_train[y_train < 0] = 1
            y_test[y_test < 0] = 1
            print("####")
            print("y_train:")
            print("Before pp:")
            print(
                "min: ",
                y_train.min(),
                " max: ",
                y_train.max(),
                " mean: ",
                y_train.mean(),
                " sd: ",
                y_train.std(),
            )
            #
            pp_scaler = StandardScaler()
            y_train = pp_scaler.fit_transform(y_train)

            print("After pp:")
            print(
                "min: ",
                y_train.min(),
                " max: ",
                y_train.max(),
                " mean: ",
                y_train.mean(),
                " sd: ",
                y_train.std(),
            )

            print("####")
            print("y_test:")
            print("Before pp:")
            print(
                "min: ",
                y_test.min(),
                " max: ",
                y_test.max(),
                " mean: ",
                y_test.mean(),
                " sd: ",
                y_test.std(),
            )
            y_test = pp_scaler.transform(y_test)
            print("After pp:")
            print(
                "min: ",
                y_test.min(),
                " max: ",
                y_test.max(),
                " mean: ",
                y_test.mean(),
                " sd: ",
                y_test.std(),
            )
            y_combined = np.concatenate((y_train, y_test), axis=0)
            df_auc_scaled = pd.DataFrame(y_combined)
            df_auc_scaled.columns = df_auc.columns
            df_auc_scaled.index = df_auc.index
            df_auc = df_auc_scaled

        y_df = df_auc[df_auc.index.isin(required_cell_line_ids)].copy()

        # For training dataset, filter for top k and bottom k drugs for each cell-line
        if self.is_train and use_k_best_worst:
            top_k_drugs_per_cell_line = {}
            y_df.replace(to_replace=-99999.0, value=1, inplace=True)

            for idx, row in y_df.iterrows():

                top_k_drugs = {}
                elements_added = 0
                for sorted_drug_idx in np.argsort(row):
                    drug_name = y_df.columns[sorted_drug_idx]
                    audrc = row[sorted_drug_idx]

                    # Consider only those AUDRC values that are < 0.25 to be "Best".
                    # If there are no such values, break
                    if (audrc != 1) and (audrc > 0.25):
                        break

                    if audrc != 1:
                        top_k_drugs[drug_name] = audrc
                        elements_added += 1

                    if elements_added == use_k_best_worst:
                        break

                top_k_drugs_per_cell_line[idx] = top_k_drugs

                bottom_k_drugs = {}
                elements_added = 0
                for sorted_drug_idx in reversed(np.argsort(row)):
                    drug_name = y_df.columns[sorted_drug_idx]
                    audrc = row[sorted_drug_idx]

                    # Consider only those AUDRC values that are > 0.75 to be "Worst".
                    # If there are no such values, break
                    if (audrc != 1) and (audrc < 0.75):
                        break

                    if audrc != 1:
                        bottom_k_drugs[drug_name] = audrc
                        elements_added += 1

                    if elements_added == use_k_best_worst:
                        break

                top_k_drugs_per_cell_line[idx].update(bottom_k_drugs)

            y_df = pd.DataFrame.from_dict(top_k_drugs_per_cell_line, orient="index")
            y_df.index.name = "depmap_id"
            y_df.fillna(-99999, inplace=True)

        # The below filter is to remove those cellines for which there are no
        # annotation features available (likely due to the absence of point
        # mutations in such cases)
        #
        # TODO: Check how to represent those cases that do not have any point mutations
        y_df = y_df[y_df.index.isin(self.df_reprn_mut.index.get_level_values(0))].copy()

        y_df = y_df.reset_index().melt(
            id_vars=[
                "depmap_id",
            ],
            var_name="drug_name",
            value_name="auc",
        )

        # When scaling is not done, filter those entries with value -99999
        self.y_df = y_df if scale_y else y_df[~(y_df.auc < 0)]

    def __len__(self):
        return len(self.y_df)

    def __getitem__(self, idx):
        record = self.y_df.iloc[idx]

        return {
            "depmap_id": record["depmap_id"],
            "drug_name": record["drug_name"],
            "auc": record["auc"],
        }

    @cached_property
    def mutations(self):
        return self.df_reprn_mut

    @cached_property
    def raw_mutations(self):
        df_reprn_mut = pd.read_csv(
            "../data/processed/ccle_raw_mutation.csv",
        )
        df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in df_reprn_mut.columns:
            mask = mask & (df_reprn_mut[col] == 0)

        df_reprn_mut = df_reprn_mut[~mask].copy()
        df_reprn_mut = df_reprn_mut.reindex(columns=GENES_324)
        return df_reprn_mut

    @cached_property
    def cnv(self):
        cnv_df = pd.read_csv(
            "../data/processed/ccle_cnv_binary.csv",
        )
        cnv_df.set_index(["depmap_id"], inplace=True)
        cnv_df = cnv_df.reindex(columns=GENES_324)
        return cnv_df

class CategoricalAnnotatedPdxDataset(PdxDataset):
    """
    PDX data, used only for testing
    """

    entity_identifier_name = "Sample"

    def __init__(
        self,
        apply_train_test_filter=True,
        is_train=False,
        only_cat_one_drugs=True,
        include_all_cell_line_drugs=False,
    ):
        # PDX#
        pdx_response = pd.read_csv("../data/pdx_drug_response.csv")
        self.is_train = is_train

        # TODO: Find smile strings for the missing drugs and update the csv
        drug_smiles = csv.reader(open(DRUG_SMILES_PATH))
        drug_names = [item[0] for item in drug_smiles]
        pdx_response = pdx_response[pdx_response["Treatment"].isin(drug_names)]

        pdx_mutation = pd.read_csv(
            "../data/processed/pdx_anno_features_only_categorical_agg.csv"
        )
        pdx_mutation.set_index("Sample", inplace=True)
        self.pdx_mutation_filtered = pdx_mutation

        pdx_response = pdx_response[
            pdx_response["Sample"].isin(pdx_mutation.index.get_level_values(0))
        ]
        self.pdx_response = pdx_response[["Sample", "Treatment", "response"]].copy()
        self.pdx_response.rename(columns={"Treatment": "drug_name"}, inplace=True)

        # Filter for category one drug, if needed
        self.only_cat_one_drugs = only_cat_one_drugs
        if self.only_cat_one_drugs:
            fname_drugs_cat = "../data/druid-druglist.csv"
            df_drugs_cat = pd.read_csv(fname_drugs_cat)
            list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])][
                "drug_name"
            ].tolist()
            self.pdx_response = self.pdx_response[
                self.pdx_response.drug_name.isin(list_drugs_cat1)
            ].reset_index(drop=True)

        if include_all_cell_line_drugs:
            if self.only_cat_one_drugs:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_CAT_1
            else:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_ALL

            pdx_response_with_required_drugs = pd.DataFrame(
                list(
                    product(
                        self.pdx_response[self.entity_identifier_name].unique(),
                        cell_line_drugs_to_use,
                    )
                ),
                columns=[self.entity_identifier_name, "drug_name"],
            )
            pdx_response_with_required_drugs = pdx_response_with_required_drugs.merge(
                self.pdx_response, how="left"
            )
            self.pdx_response = pdx_response_with_required_drugs

        if apply_train_test_filter:
            uniq_sample_ids = self.pdx_response.Sample.unique()
            train_ids, test_ids, _, _ = train_test_split(
                uniq_sample_ids,
                np.arange(len(uniq_sample_ids)),
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
            filter_ids = train_ids if is_train else test_ids
            self.pdx_response = self.pdx_response[
                self.pdx_response.Sample.isin(filter_ids)
            ].copy()

    def __len__(self):
        return len(self.pdx_response)

    def __getitem__(self, idx):
        record = self.pdx_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
        }

    @cached_property
    def mutations(self):
        return self.pdx_mutation_filtered

    @cached_property
    def raw_mutations(self):
        pdx_mutation = pd.read_csv("../data/processed/pdx_mutations.csv")
        pdx_mutation.set_index("Sample", inplace=True)
        pdx_mutation = pdx_mutation.reindex(columns=GENES_324)
        return pdx_mutation

    @cached_property
    def cnv(self):
        cnv_df = pd.read_csv(
            "../data/processed/pdx_cnv.csv",
        )
        cnv_df.set_index(["Sample"], inplace=True)
        cnv_df = cnv_df.reindex(columns=GENES_324)
        # CNV represents no change/loss/amplification with 0/-1/1. For the purpose
        # of distance computation, it is sufficient if they are only
        # indicator variables (0/1) and the below does the same
        cnv_df.replace(to_replace=-1, value=1, inplace=True)
        return cnv_df

class TcgaDataset(Dataset):
    """
    Base class for datasets that hold TCGA information
    """

    base_dir = DATA_BASE_DIR

    def __str__(self):
        dataset_df = pd.concat(list(self[: len(self)].values()), axis=1)
        return f"""{self.__class__.__name__} {'Train' if self.is_train else 'Test'} Set
        #Entities - {len(dataset_df[self.entity_identifier_name].unique())}
        #Drugs - {len(dataset_df.drug_name.unique())}
        #Pairs - {len(self)}
        #Response (0 to 1) - {dataset_df.response.value_counts()[0]} to {dataset_df.response.value_counts()[1]}
        """

    pass

class CategoricalAnnotatedTcgaDataset(TcgaDataset):
    """
    TCGA data, used only for testing
    """

    entity_identifier_name = "submitter_id"

    def __init__(
        self,
        apply_train_test_filter=True,
        is_train=False,
        only_cat_one_drugs=True,
        include_all_cell_line_drugs=False,
    ):
        self.is_train = is_train
        tcga_response = pd.read_csv("../data/processed/TCGA_drug_response_010222.csv")
        tcga_response.rename(
            columns={
                "patient.arr": self.entity_identifier_name,
                "drug": "drug_name",
                "response": "response_description",
                "response_cat": "response",
            },
            inplace=True,
        )

        # TODO: Find smile strings for the missing drugs and update the csv
        drug_smiles = csv.reader(open(DRUG_SMILES_PATH))

        drug_names = [item[0] for item in drug_smiles]
        tcga_response = tcga_response[tcga_response["drug_name"].isin(drug_names)]

        tcga_mutation = pd.read_csv(
            "../data/processed/tcga_anno_features_only_categorical_agg.csv"
        )
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        self.tcga_mutation_filtered = tcga_mutation

        tcga_response = tcga_response[
            tcga_response[self.entity_identifier_name].isin(
                tcga_mutation.index.get_level_values(0)
            )
        ]
        self.tcga_response = tcga_response[
            [self.entity_identifier_name, "drug_name", "response"]
        ].copy()

        self.only_cat_one_drugs = only_cat_one_drugs
        if self.only_cat_one_drugs:
            fname_drugs_cat = "../data/druid-druglist.csv"
            df_drugs_cat = pd.read_csv(fname_drugs_cat)
            list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])][
                "drug_name"
            ].tolist()
            self.tcga_response = self.tcga_response[
                self.tcga_response.drug_name.isin(list_drugs_cat1)
            ].reset_index(drop=True)

        if include_all_cell_line_drugs:
            if self.only_cat_one_drugs:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_CAT_1
            else:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_ALL

            tcga_response_with_required_drugs = pd.DataFrame(
                list(
                    product(
                        self.tcga_response[self.entity_identifier_name].unique(),
                        cell_line_drugs_to_use,
                    )
                ),
                columns=[self.entity_identifier_name, "drug_name"],
            )
            tcga_response_with_required_drugs = tcga_response_with_required_drugs.merge(
                self.tcga_response, how="left"
            )
            self.tcga_response = tcga_response_with_required_drugs

        if apply_train_test_filter:

            uniq_submitter_ids = self.tcga_response.submitter_id.unique()
            train_ids, test_ids, _, _ = train_test_split(
                uniq_submitter_ids,
                np.arange(len(uniq_submitter_ids)),
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
            filter_ids = train_ids if is_train else test_ids
            self.tcga_response = self.tcga_response[
                self.tcga_response.submitter_id.isin(filter_ids)
            ].copy()

    def __len__(self):
        return len(self.tcga_response)

    def __getitem__(self, idx):
        record = self.tcga_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
        }

    @cached_property
    def mutations(self):
        return self.tcga_mutation_filtered

    @cached_property
    def raw_mutations(self):
        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        tcga_mutation = tcga_mutation.reindex(columns=GENES_324)
        return tcga_mutation

    @cached_property
    def cnv(self):
        tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_010222")
        tcga_cnv.rename(columns={"Unnamed: 0": "submitter_id"}, inplace=True)
        tcga_cnv.set_index("submitter_id", inplace=True)
        tcga_cnv = tcga_cnv.reindex(columns=GENES_324)
        return tcga_cnv

    @cached_property
    def survival_info(self):
        survival_info_df = pd.read_csv("../data/processed/survival_rate_final_010222")
        survival_info_df.rename(
            columns={"demographic.days_to_death": "days"}, inplace=True
        )
        survival_info_df.drop(
            columns=["demographic.vital_status", "days_to_death_scaled"], inplace=True
        )
        return survival_info_df

    @cached_property
    def gene_exp(self):
        # change the file path to the tcga one
        gene_exp_df = pd.read_csv("../data/processed/tcga_gene_expression.csv")
        gene_exp_df.rename(columns={"tcga_id": "submitter_id"}, inplace=True)
        gene_exp_df.drop_duplicates(subset=["submitter_id"], inplace=True) # The same TCGA patient had multiple gene expression values - this takes just the first entry
        gene_exp_df.set_index("submitter_id", drop=True, inplace=True)
        return gene_exp_df[GENES_324]



class AggCategoricalAnnotatedCellLineDataset(CategoricalAnnotatedCellLineDataset):
    """
    Cell line data with categorical annotation features from annovar aggregated per gene
    """

    @cached_property
    def mutations(self):
        return agg_anno_features(self.df_reprn_mut)

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df

class AggCategoricalAnnotatedPdxDataset(CategoricalAnnotatedPdxDataset):
    """
    Aggregated categorical annotations features for PDX entities
    """

    @cached_property
    def mutations(self):
        return agg_anno_features(self.pdx_mutation_filtered)

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df


class AggCategoricalAnnotatedTcgaDataset(CategoricalAnnotatedTcgaDataset):
    """
    Aggregated categorical annotations features for TCGA entities
    """

    @cached_property
    def mutations(self):
        return agg_anno_features(self.tcga_mutation_filtered)

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df


class RawCellLineDataset(CellLineDataset):
    """
    Cell line data with raw mutation feature
    """

    entity_identifier_name = "depmap_id"

    def get_df_data_tfidf(self, df_data):
        col = df_data.values
        mx = np.ma.masked_where(col != -99999, col)
        mask = np.ma.getmask(mx)
        col_sum = np.sum(col, where=mask)
        TF = col / col_sum
        #     TF = np.divide(col, col_sum, where=mask)
        #     print(len(TF))
        N = np.sum(mask)
        col_mean = np.mean(col, where=mask)
        col_bin = col >= col_mean
        DF = col_bin.sum(0)
        IDF = np.log(N / 1 + DF)
        df_data_tfidf = TF * IDF
        # print(N)
        #     assert not df_data_tfidf.isnull().any().any()
        return df_data_tfidf

    def __init__(
        self,
        is_train=True,
        only_cat_one_drugs=True,
        scale_y=False,
        use_k_best_worst=None,
        use_tf_idf_scaling=False,
    ):
        """
        Parameters
        ----------
        is_train : bool
            Returns items from the train or test split (defaults to True)
        only_cat_one_drugs : bool
            Filters for items corresponding to category 1 drugs (defaults to True)

        """
        self.is_train = is_train
        self.df_reprn_mut = pd.read_csv(
            "../data/processed/ccle_raw_mutation.csv",
        )
        self.df_reprn_mut.set_index(["depmap_id"], inplace=True)

        # Check if there are any cell-lines for which there are no valid mutations and
        # filter those out
        mask = True
        for col in self.df_reprn_mut.columns:
            mask = mask & (self.df_reprn_mut[col] == 0)

        self.df_reprn_mut = self.df_reprn_mut[~mask].copy()

        df_auc = pd.read_csv("../data/cell_drug_auc_final_1111.csv")
        df_auc["depmap_id"] = df_auc["ARXSPAN_ID"].astype("string")
        df_auc.drop("ARXSPAN_ID", axis=1, inplace=True)
        df_auc.set_index(["depmap_id"], inplace=True)

        if use_tf_idf_scaling:
            df_auc = df_auc.apply(self.get_df_data_tfidf, axis=0)
            nan_locations = np.where((df_auc.values < 0) & (df_auc.values != -99999))
            # Replace all highly negative values on the tfidf columns with Nan
            for i, j in zip(nan_locations[0], nan_locations[1]):
                df_auc.iloc[i, j] = np.nan

        # Filter for category one drug, if needed
        self.only_cat_one_drugs = only_cat_one_drugs
        if self.only_cat_one_drugs:
            fname_drugs_cat = "../data/druid-druglist.csv"
            df_drugs_cat = pd.read_csv(fname_drugs_cat)
            list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])][
                "drug_name"
            ].tolist()

            # The below two drugs are effective in a lot of cell-lines which skews the
            # predictions considerably. Hence, we remove these two drugs from training
            # if tf-idf scaling is not done
            if not use_tf_idf_scaling:
                list_drugs_cat1.remove("PAZOPANIB")
                list_drugs_cat1.remove("IMATINIB")

            filtered_drugs = [col for col in df_auc.columns if col in list_drugs_cat1]
            df_auc = df_auc[filtered_drugs]

        train_cell_lines_ids = pd.read_csv(
            "../data/train_celllines_v1_1111", header=None
        )[0].values

        test_cell_lines_ids = pd.read_csv(
            "../data/test_celllines_v1_1111", header=None
        )[0].values

        if is_train is not None:
            if self.is_train:
                required_cell_line_ids = train_cell_lines_ids
            else:
                required_cell_line_ids = test_cell_lines_ids
        else:
            required_cell_line_ids = np.concatenate(
                [train_cell_lines_ids, test_cell_lines_ids]
            )

        if scale_y:
            y_train_df = df_auc[df_auc.index.isin(train_cell_lines_ids)].copy()
            y_train = y_train_df.to_numpy()
            y_test_df = df_auc[df_auc.index.isin(test_cell_lines_ids)].copy()
            y_test = y_test_df.to_numpy()
            y_train[y_train < 0] = 1
            y_test[y_test < 0] = 1
            print("####")
            print("y_train:")
            print("Before pp:")
            print(
                "min: ",
                y_train.min(),
                " max: ",
                y_train.max(),
                " mean: ",
                y_train.mean(),
                " sd: ",
                y_train.std(),
            )
            #
            pp_scaler = StandardScaler()
            y_train = pp_scaler.fit_transform(y_train)

            print("After pp:")
            print(
                "min: ",
                y_train.min(),
                " max: ",
                y_train.max(),
                " mean: ",
                y_train.mean(),
                " sd: ",
                y_train.std(),
            )

            print("####")
            print("y_test:")
            print("Before pp:")
            print(
                "min: ",
                y_test.min(),
                " max: ",
                y_test.max(),
                " mean: ",
                y_test.mean(),
                " sd: ",
                y_test.std(),
            )
            y_test = pp_scaler.transform(y_test)
            print("After pp:")
            print(
                "min: ",
                y_test.min(),
                " max: ",
                y_test.max(),
                " mean: ",
                y_test.mean(),
                " sd: ",
                y_test.std(),
            )
            y_combined = np.concatenate((y_train, y_test), axis=0)
            df_auc_scaled = pd.DataFrame(y_combined)
            df_auc_scaled.columns = df_auc.columns
            df_auc_scaled.index = df_auc.index
            df_auc = df_auc_scaled

        y_df = df_auc[df_auc.index.isin(required_cell_line_ids)].copy()

        # For training dataset, filter for top k and bottom k drugs for each cell-line
        if (self.is_train is True) and use_k_best_worst:
            top_k_drugs_per_cell_line = {}
            y_df.replace(to_replace=-99999.0, value=1, inplace=True)

            for idx, row in y_df.iterrows():

                top_k_drugs = {}
                elements_added = 0
                for sorted_drug_idx in np.argsort(row):
                    drug_name = y_df.columns[sorted_drug_idx]
                    audrc = row[sorted_drug_idx]

                    # Consider only those AUDRC values that are < 0.25 to be "Best".
                    # If there are no such values, break
                    if (audrc != 1) and (audrc > 0.25):
                        break

                    if audrc != 1:
                        top_k_drugs[drug_name] = audrc
                        elements_added += 1

                    if elements_added == use_k_best_worst:
                        break

                top_k_drugs_per_cell_line[idx] = top_k_drugs

                bottom_k_drugs = {}
                elements_added = 0
                for sorted_drug_idx in reversed(np.argsort(row)):
                    drug_name = y_df.columns[sorted_drug_idx]
                    audrc = row[sorted_drug_idx]

                    # Consider only those AUDRC values that are > 0.75 to be "Worst".
                    # If there are no such values, break
                    if (audrc != 1) and (audrc < 0.75):
                        break

                    if audrc != 1:
                        bottom_k_drugs[drug_name] = audrc
                        elements_added += 1

                    if elements_added == use_k_best_worst:
                        break

                top_k_drugs_per_cell_line[idx].update(bottom_k_drugs)

            y_df = pd.DataFrame.from_dict(top_k_drugs_per_cell_line, orient="index")
            y_df.index.name = "depmap_id"
            y_df.fillna(-99999, inplace=True)

        # The below filter is to remove those cellines for which there are no
        # annotation features available (likely due to the absence of point
        # mutations in such cases)
        #
        # TODO: Check how to represent those cases that do not have any point mutations
        y_df = y_df[y_df.index.isin(self.df_reprn_mut.index.get_level_values(0))].copy()

        y_df = y_df.reset_index().melt(
            id_vars=[
                "depmap_id",
            ],
            var_name="drug_name",
            value_name="auc",
        )

        # When scaling is not done, filter those entries with value -99999
        self.y_df = y_df if scale_y else y_df[~(y_df.auc < 0)]

    def __len__(self):
        return len(self.y_df)

    def __getitem__(self, idx):
        record = self.y_df.iloc[idx]

        return {
            "depmap_id": record["depmap_id"],
            "drug_name": record["drug_name"],
            "auc": record["auc"],
        }

    @cached_property
    def mutations(self):
        return self.df_reprn_mut

    @cached_property
    def cnv(self):
        cnv_df = pd.read_csv(
            "../data/processed/ccle_cnv_binary.csv",
        )
        cnv_df.set_index(["depmap_id"], inplace=True)
        cnv_df = cnv_df.reindex(columns=GENES_324)
        # CNV represents no change/loss/amplification with 0/-1/1. For the purpose
        # of distance computation, it is sufficient if they are only
        # indicator variables (0/1) and the below does the same
        cnv_df.replace(to_replace=-1, value=1, inplace=True)
        return cnv_df

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df

class RawPdxDataset(PdxDataset):
    """
    PDX data with raw mutation features and drug fingerprints
    """

    entity_identifier_name = "Sample"

    def __init__(
        self, is_train=False, only_cat_one_drugs=True, include_all_cell_line_drugs=False
    ):
        # PDX#
        pdx_response = pd.read_csv("../data/pdx_drug_response.csv")
        self.is_train = is_train

        pdx_mutation = pd.read_csv("../data/processed/pdx_mutations.csv")
        pdx_mutation.set_index("Sample", inplace=True)
        self.pdx_mutation_filtered = pdx_mutation
        pdx_response = pdx_response[
            pdx_response["Sample"].isin(pdx_mutation.index.get_level_values(0))
        ]
        self.pdx_response = pdx_response[["Sample", "Treatment", "response"]].copy()
        self.pdx_response.rename(columns={"Treatment": "drug_name"}, inplace=True)

        # Filter for category one drug, if needed
        self.only_cat_one_drugs = only_cat_one_drugs
        if self.only_cat_one_drugs:
            fname_drugs_cat = "../data/druid-druglist.csv"
            df_drugs_cat = pd.read_csv(fname_drugs_cat)
            list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])][
                "drug_name"
            ].tolist()
            self.pdx_response = self.pdx_response[
                self.pdx_response.drug_name.isin(list_drugs_cat1)
            ].reset_index(drop=True)

        if include_all_cell_line_drugs:
            if self.only_cat_one_drugs:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_CAT_1
            else:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_ALL

            pdx_response_with_required_drugs = pd.DataFrame(
                list(
                    product(
                        self.pdx_response[self.entity_identifier_name].unique(),
                        cell_line_drugs_to_use,
                    )
                ),
                columns=[self.entity_identifier_name, "drug_name"],
            )
            pdx_response_with_required_drugs = pdx_response_with_required_drugs.merge(
                self.pdx_response, how="left"
            )
            self.pdx_response = pdx_response_with_required_drugs

        if is_train is not None:
            uniq_sample_ids = self.pdx_response.Sample.unique()
            train_ids, test_ids, _, _ = train_test_split(
                uniq_sample_ids,
                np.arange(len(uniq_sample_ids)),
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
            filter_ids = train_ids if is_train else test_ids
            self.pdx_response = self.pdx_response[
                self.pdx_response.Sample.isin(filter_ids)
            ].copy()

    def __len__(self):
        return len(self.pdx_response)

    def __getitem__(self, idx):
        record = self.pdx_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
        }

    @cached_property
    def mutations(self):
        return self.pdx_mutation_filtered

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df

    @cached_property
    def cnv(self):
        cnv_df = pd.read_csv(
            "../data/processed/pdx_cnv.csv",
        )
        cnv_df.set_index(["Sample"], inplace=True)
        cnv_df = cnv_df.reindex(columns=GENES_324)
        # CNV represents no change/loss/amplification with 0/-1/1. For the purpose
        # of distance computation, it is sufficient if they are only
        # indicator variables (0/1) and the below does the same
        cnv_df.replace(to_replace=-1, value=1, inplace=True)
        return cnv_df


class RawTcgaDataset(TcgaDataset):
    """
    TCGA data
    """

    entity_identifier_name = "submitter_id"

    def __init__(
        self, is_train=False, only_cat_one_drugs=True, include_all_cell_line_drugs=False
    ):
        self.is_train = is_train
        tcga_response = pd.read_csv("../data/processed/TCGA_drug_response_010222.csv")
        tcga_response.rename(
            columns={
                "patient.arr": self.entity_identifier_name,
                "drug": "drug_name",
                "response": "response_description",
                "response_cat": "response",
            },
            inplace=True,
        )

        tcga_mutation = pd.read_csv("../data/processed/tcga_mut_final_barcode_010222")
        tcga_mutation.set_index(self.entity_identifier_name, inplace=True)
        self.tcga_mutation_filtered = tcga_mutation

        tcga_response = tcga_response[
            tcga_response[self.entity_identifier_name].isin(
                tcga_mutation.index.get_level_values(0)
            )
        ]
        self.tcga_response = tcga_response[
            [self.entity_identifier_name, "drug_name", "response"]
        ].copy()

        self.only_cat_one_drugs = only_cat_one_drugs
        if self.only_cat_one_drugs:
            fname_drugs_cat = "./../../data/drugs/druid-druglist.csv"
            df_drugs_cat = pd.read_csv(fname_drugs_cat)
            list_drugs_cat1 = df_drugs_cat[df_drugs_cat["category"].isin([1])][
                "drug_name"
            ].tolist()
            self.tcga_response = self.tcga_response[
                self.tcga_response.drug_name.isin(list_drugs_cat1)
            ].reset_index(drop=True)

        if include_all_cell_line_drugs:
            if self.only_cat_one_drugs:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_CAT_1
            else:
                cell_line_drugs_to_use = CELL_LINE_DRUGS_ALL

            tcga_response_with_required_drugs = pd.DataFrame(
                list(
                    product(
                        self.tcga_response[self.entity_identifier_name].unique(),
                        cell_line_drugs_to_use,
                    )
                ),
                columns=[self.entity_identifier_name, "drug_name"],
            )
            tcga_response_with_required_drugs = tcga_response_with_required_drugs.merge(
                self.tcga_response, how="left"
            )
            self.tcga_response = tcga_response_with_required_drugs

        if is_train is not None:
            uniq_submitter_ids = self.tcga_response.submitter_id.unique()
            train_ids, test_ids, _, _ = train_test_split(
                uniq_submitter_ids,
                np.arange(len(uniq_submitter_ids)),
                test_size=0.2,
                random_state=RANDOM_STATE,
            )
            filter_ids = train_ids if is_train else test_ids
            self.tcga_response = self.tcga_response[
                self.tcga_response.submitter_id.isin(filter_ids)
            ].copy()

    def __len__(self):
        return len(self.tcga_response)

    def __getitem__(self, idx):
        record = self.tcga_response.iloc[idx]

        return {
            self.entity_identifier_name: record[self.entity_identifier_name],
            "drug_name": record["drug_name"],
            "response": record["response"],
        }

    @cached_property
    def mutations(self):
        return self.tcga_mutation_filtered

    @cached_property
    def cnv(self):
        tcga_cnv = pd.read_csv("../data/processed/tcga_cnv_final_barcode_010222")
        tcga_cnv.rename(columns={"Unnamed: 0": "submitter_id"}, inplace=True)
        tcga_cnv.set_index("submitter_id", inplace=True)
        tcga_cnv = tcga_cnv.reindex(columns=GENES_324)
        return tcga_cnv

    @cached_property
    def drug_repr(self):
        drug_fp_df = pd.read_csv("../data/processed/drug_morgan_fingerprints.csv")
        drug_fp_df.set_index("drug_name", inplace=True)
        return drug_fp_df

    @cached_property
    def survival_info(self):
        survival_info_df = pd.read_csv("../data/processed/survival_rate_final_010222")
        survival_info_df.rename(
            columns={"demographic.days_to_death": "days"}, inplace=True
        )
        survival_info_df.drop(
            columns=["demographic.vital_status", "days_to_death_scaled"], inplace=True
        )
        return survival_info_df
