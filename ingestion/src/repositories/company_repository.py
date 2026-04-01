from sqlalchemy import text
from sqlalchemy.orm import Session

from models import CompanyUpsert


class CompanyRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, company: CompanyUpsert) -> str:
        sql = text("""
            INSERT INTO market_data.companies (
                company_name, trading_name, sector, subsector, segment, description, website, cnpj, updated_at
            ) VALUES (
                :company_name, :trading_name, :sector, :subsector, :segment, :description, :website, :cnpj, NOW()
            )
            ON CONFLICT (company_name)
            DO UPDATE SET
                trading_name = COALESCE(EXCLUDED.trading_name, market_data.companies.trading_name),
                sector = COALESCE(EXCLUDED.sector, market_data.companies.sector),
                subsector = COALESCE(EXCLUDED.subsector, market_data.companies.subsector),
                segment = COALESCE(EXCLUDED.segment, market_data.companies.segment),
                description = COALESCE(EXCLUDED.description, market_data.companies.description),
                website = COALESCE(EXCLUDED.website, market_data.companies.website),
                cnpj = COALESCE(EXCLUDED.cnpj, market_data.companies.cnpj),
                updated_at = NOW()
            RETURNING id
        """)
        row = self.session.execute(sql, company.__dict__).fetchone()
        return str(row[0])

    def update_company_by_id(self, company_id: str, company: CompanyUpsert) -> None:
        sql = text("""
            UPDATE market_data.companies
            SET
                trading_name = COALESCE(:trading_name, trading_name),
                sector = COALESCE(:sector, sector),
                subsector = COALESCE(:subsector, subsector),
                segment = COALESCE(:segment, segment),
                description = COALESCE(:description, description),
                website = COALESCE(:website, website),
                cnpj = COALESCE(:cnpj, cnpj),
                updated_at = NOW()
            WHERE id = :company_id
        """)
        payload = {
            "company_id": company_id,
            **company.__dict__,
        }
        self.session.execute(sql, payload)