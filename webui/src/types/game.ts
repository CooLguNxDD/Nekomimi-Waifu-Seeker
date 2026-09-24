export interface CharacterCard {
  id: string;
  name: string;
  series?: string;
  medium?: string;
  blurb?: string;
  image_url?: string | null;
  source_url?: string | null;
  confidence?: number;
  probability?: number;
  fit_score?: number | null;
  tags?: string[];
}

export interface SearchRound {
  round?: number;
  query?: string;
  raw_hits?: number;
  new_candidates?: number;
  error?: string;
}

export interface DetermineResult {
  mode?: string;
  query?: string;
  winner: CharacterCard | null;
  runners_up?: CharacterCard[];
  notes?: string;
  search?: {
    online_used?: boolean;
    online_count?: number;
    catalog_size?: number;
    online_error?: string;
    rounds?: SearchRound[];
  };
  error?: string;
}

export interface QuestionOption {
  key: string;
  label: string;
}

export interface NekomimiQuestion {
  qid: string;
  text: string;
  category?: string;
  kind?: string;
  turn?: number;
  max_turns?: number;
  options?: QuestionOption[];
}

export interface NekomimiState {
  session_id?: string;
  stage?: "asking" | "guessing" | "done" | string;
  question?: NekomimiQuestion;
  top?: CharacterCard[];
  candidates_alive?: number;
  laya?: boolean;
  guess?: CharacterCard | null;
  guess_number?: number;
  message?: string;
  correct?: boolean;
  winner?: CharacterCard | null;
  turns?: number;
  seed?: string;
  error?: string;
}
